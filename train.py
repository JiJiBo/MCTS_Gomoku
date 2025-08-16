import argparse
import os
import random
import threading
import queue
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time  # 为了添加日志时间戳

from core.board import GomokuBoard
from mcts.MCTS import MCTS
from net.GomokuNet import PolicyValueNet


def selfplay_worker(
        model: PolicyValueNet,
        data_queue: queue.Queue,
        stop_event: threading.Event,
        model_lock: threading.Lock,
        num_simulations: int,
        opponent_type: str = "self",
        opponent_simulations: int | None = None,
):
    """Generate training data by playing against a weaker opponent."""
    local_model = PolicyValueNet(board_size=model.H)
    opponent_simulations = opponent_simulations or max(1, num_simulations // 10)
    while not stop_event.is_set():
        # Refresh local model parameters
        with model_lock:
            local_model.load_state_dict(model.state_dict())
        strong_mcts = MCTS(local_model)
        weak_mcts = MCTS(local_model) if opponent_type == "weak_mcts" else None
        board = GomokuBoard(size=model.H)
        player = 1
        while not board.is_terminal() and not stop_event.is_set():
            if player == 1 or opponent_type == "self":
                _, probs = strong_mcts.run(
                    board, player, number_samples=num_simulations, is_train=True
                )
                flat = probs.reshape(-1)
                move = np.random.choice(len(flat), p=flat)
            elif opponent_type == "random":
                legal = board.legal_moves()
                move = random.choice(legal)
                y, x = move
                board.step((y, x), player)
                player = -player
                continue
            else:  # weak_mcts
                _, probs = weak_mcts.run(
                    board, player, number_samples=opponent_simulations, is_train=False
                )
                flat = probs.reshape(-1)
                move = np.random.choice(len(flat), p=flat)
            y, x = move if isinstance(move, tuple) else divmod(move, board.size)
            board.step((y, x), player)
            player = -player
        result = board.winner()
        boards, policies, values, _ = strong_mcts.get_train_data(game_result=result)
        for b, p, v in zip(boards, policies, values):
            data_queue.put((b, p.view(-1), v))


def train(args):
    seed = 42
    print("Random Seed: ", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():  # ✅ 修复：仅在可用时设 CUDA 种子
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # ---- checkpoints path ----
    os.makedirs('./check_dir', exist_ok=True)
    run_id = len(os.listdir('./check_dir'))
    checkpoints_path = f"./check_dir/run{run_id}"
    os.makedirs(checkpoints_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model = PolicyValueNet(board_size=args.board_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    log_dir = args.log_dir
    log_dir = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)

    data_queue: queue.Queue = queue.Queue(maxsize=args.queue_size)
    stop_event = threading.Event()
    model_lock = threading.Lock()

    selfplay_worker(
        model,
        data_queue,
        stop_event,
        model_lock,
        args.num_simulations,
        args.opponent_type,
        args.opponent_simulations,
    )

    global_step = 0
    try:
        for step in range(args.train_steps):
            batch = []
            while len(batch) < args.batch_size:
                try:
                    batch.append(data_queue.get(timeout=1))
                except queue.Empty:
                    if stop_event.is_set():
                        break
            if len(batch) < args.batch_size:
                print(f"[Train Step {step}] Batch is too small, exiting training.")
                break

            boards = torch.stack([b for b, _, _ in batch]).to(device)
            policies = torch.stack([p for _, p, _ in batch]).to(device)
            values = torch.tensor([v for _, _, v in batch], dtype=torch.float32).view(-1, 1).to(device)

            # 训练日志
            print(f"[Train Step {step}] Training on batch of size {len(batch)}.")

            with model_lock:
                pred_pi, pred_v = model(boards)
                policy_loss = -(policies * torch.log(pred_pi + 1e-8)).sum(dim=1).mean()
                value_loss = F.mse_loss(pred_v, values)
                loss = policy_loss + value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 写入日志
            writer.add_scalar('loss/policy', policy_loss.item(), global_step)
            writer.add_scalar('loss/value', value_loss.item(), global_step)
            writer.add_scalar('loss/total', loss.item(), global_step)
            if step % args.save_interval == 0:
                mem_path = os.path.join(checkpoints_path, f'mem{step}.pth')
                torch.save(model.state_dict(), mem_path)
            # 打印训练信息
            print(
                f"[Train Step {step}] Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Total Loss: {loss.item():.4f}")
            global_step += 1

    finally:
        stop_event.set()
        torch.save(model.state_dict(), args.save_path)
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time training with self-play MCTS data')
    parser.add_argument('--board-size', type=int, default=15)
    parser.add_argument('--num-simulations', type=int, default=800)  # MCTS 深度更大
    parser.add_argument('--opponent-type', type=str, choices=['self', 'random', 'weak_mcts'], default='self',
                        help='type of opponent for training: self, random, or weak_mcts')
    parser.add_argument('--opponent-simulations', type=int, default=10,
                        help='number of simulations for weak_mcts opponent')
    parser.add_argument('--train-steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=512)  # 3090 显存足够
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--queue-size', type=int, default=2048)  # 较大的自对弈数据缓存
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log-dir', type=str, default='/root/tf-logs/')
    parser.add_argument('--save-path', type=str, default='realtime_model.pth')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA')
    args = parser.parse_args()

    # 开始训练前输出开始时间
    print(f"[Training Start] Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    train(args)
    # 训练完成时输出结束时间
    print(f"[Training End] Training ended at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
