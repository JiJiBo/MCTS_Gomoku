import argparse
import os
import random
import threading
import queue
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import tqdm

from core.board import GomokuBoard
from mcts.MCTS import MCTS
from net.GomokuNet import PolicyValueNet


def safe_choice(probs):
    # 替换 NaN 为 0
    probs = np.nan_to_num(probs, nan=0.0)
    # 避免全为 0
    total = probs.sum()
    if total == 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / total
    return np.random.choice(len(probs), p=probs)


def selfplay_worker(worker_id, strong_model, weak_model, data_queue, stop_event, model_lock, num_simulations,
                    opponent_type="weak_mcts", opponent_simulations=None):
    print(f"[Worker {worker_id}] Starting self-play worker.")
    try:
        local_strong = PolicyValueNet(board_size=strong_model.H)
        local_weak = PolicyValueNet(board_size=weak_model.H) if weak_model else None
        opponent_simulations = opponent_simulations or max(1, num_simulations // 10)

        while not stop_event.is_set():
            with model_lock:
                local_strong.load_state_dict(strong_model.state_dict())
                if local_weak:
                    local_weak.load_state_dict(weak_model.state_dict())

            board = GomokuBoard(size=strong_model.H)
            player = 1

            strong_mcts = MCTS(local_strong)
            weak_mcts = MCTS(local_weak) if local_weak else None

            while not board.is_terminal() and not stop_event.is_set():
                if player == 1:
                    _, probs = strong_mcts.run(board, player, number_samples=num_simulations, is_train=True)
                else:
                    if opponent_type == "random":
                        legal = board.legal_moves()
                        move = random.choice(legal)
                        y, x = move
                        board.step((y, x), player)
                        player = -player
                        continue
                    else:
                        _, probs = weak_mcts.run(board, player, number_samples=opponent_simulations, is_train=False)

                # 确保只选择合法落子位置
                legal = board.legal_moves()
                flat_probs = probs.reshape(-1)
                # 将非法位置的概率设为0
                mask = np.zeros_like(flat_probs, dtype=bool)
                for yx in legal:
                    idx = yx[0] * board.size + yx[1]
                    mask[idx] = True
                flat_probs = np.where(mask, flat_probs, 0.0)
                # 如果概率和为0，平均分配给所有合法位置
                s = flat_probs.sum()
                if s > 0:
                    flat_probs /= s
                else:
                    flat_probs[mask] = 1.0 / len(legal)

                move_idx = safe_choice(flat_probs)
                y, x = divmod(move_idx, board.size)
                board.step((y, x), player)
                player = -player

            result = board.winner()
            boards, policies, values, _ = strong_mcts.get_train_data(game_result=result)
            for b, p, v in zip(boards, policies, values):
                data_queue.put((b, p.view(-1), v))
            print(f" data_queue size: {data_queue.qsize()}")

    except  Exception as e:
        print(f"[Worker {worker_id}] Stopping self-play worker by  {e}.")
    print(f"[Worker {worker_id}] Stopped self-play worker.")


def train_realtime(args, update_threshold=0.6):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.makedirs('./check_dir', exist_ok=True)
    run_id = len(os.listdir('./check_dir'))
    checkpoints_path = f"./check_dir/run{run_id}"
    os.makedirs(checkpoints_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    strong_model = PolicyValueNet(board_size=args.board_size).to(device)
    weak_model = PolicyValueNet(board_size=args.board_size).to(device)
    optimizer = torch.optim.Adam(strong_model.parameters(), lr=args.lr)

    log_dir = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)

    data_queue: queue.Queue = queue.Queue(maxsize=args.queue_size)
    stop_event = threading.Event()
    model_lock = threading.Lock()

    workers = []
    for i in range(args.num_workers):
        t = threading.Thread(
            target=selfplay_worker,
            args=(
                i, strong_model, weak_model, data_queue, stop_event, model_lock, args.num_simulations,
                args.opponent_type,
                args.opponent_simulations),
            daemon=True
        )
        t.start()
        workers.append(t)

    global_step = 0
    recent_results = []

    last_print_time = time.time()  # 记录上一次打印时间

    try:
        for step in tqdm.trange(args.train_steps, desc="Training"):
            batch = []
            while len(batch) < args.batch_size:
                try:
                    batch.append(data_queue.get(timeout=1))
                    current_time = time.time()
                    if current_time - last_print_time >= 300:  # 300秒 = 5分钟
                        print(f"[Train Step {step}] Current batch length: {len(batch)}")
                        last_print_time = current_time
                except queue.Empty:
                    if stop_event.is_set():
                        break

            # 如果 batch 小于 batch_size，则退出训练
            if len(batch) < args.batch_size:
                print(f"[Train Step {step}] Batch too small, exiting training.")
                break

            # 每隔5分钟打印一次 batch 长度


            boards = torch.stack([b for b, _, _ in batch]).to(device)
            policies = torch.stack([p for _, p, _ in batch]).to(device)
            values = torch.tensor([v for _, _, v in batch], dtype=torch.float32).view(-1, 1).to(device)

            print(f"[Train Step {step}] Training on batch of size {len(batch)}.")

            with model_lock:
                pred_pi, pred_v = strong_model(boards)
                policy_loss = -(policies * torch.log(pred_pi + 1e-8)).sum(dim=1).mean()
                value_loss = F.mse_loss(pred_v, values)
                loss = policy_loss + value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            writer.add_scalar('loss/policy', policy_loss.item(), global_step)
            writer.add_scalar('loss/value', value_loss.item(), global_step)
            writer.add_scalar('loss/total', loss.item(), global_step)

            # 更新弱模型
            recent_results.append((values.mean().item() > 0))
            if len(recent_results) > 100:
                recent_results.pop(0)
            win_rate = np.mean(recent_results)
            if win_rate >= update_threshold:
                with model_lock:
                    weak_model.load_state_dict(strong_model.state_dict())
                print(f"[Update] Weak model updated at step {step}, recent win rate: {win_rate:.2f}")

            if step % args.save_interval == 0:
                mem_path = os.path.join(checkpoints_path, f'mem{step}.pth')
                torch.save(strong_model.state_dict(), mem_path)

            print(
                f"[Train Step {step}] Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Total Loss: {loss.item():.4f}")
            global_step += 1

    finally:
        stop_event.set()
        for t in workers:
            t.join()
        torch.save(strong_model.state_dict(), args.save_path)
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time training with strong vs weak self-play MCTS')
    parser.add_argument('--board-size', type=int, default=15)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--num-simulations', type=int, default=800)
    parser.add_argument('--opponent-type', type=str, choices=['random', 'weak_mcts'], default='weak_mcts')
    parser.add_argument('--opponent-simulations', type=int, default=100)
    parser.add_argument('--train-steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--queue-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log-dir', type=str, default='/root/tf-logs/')
    parser.add_argument('--save-path', type=str, default='realtime_model.pth')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA')
    parser.add_argument('--update-threshold', type=float, default=0.6, help='Win rate threshold to update weak model')
    args = parser.parse_args()

    print(f"[Training Start] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    train_realtime(args, update_threshold=args.update_threshold)
    print(f"[Training End] {time.strftime('%Y-%m-%d %H:%M:%S')}")
