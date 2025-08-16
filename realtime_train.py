import argparse
import os
import random
import multiprocessing as mp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import tqdm

from core.board import GomokuBoard
from mcts.MCTS import MCTS
from net.GomokuNet import PolicyValueNet

mp.set_start_method('spawn', force=True)

def safe_choice(probs):
    probs = np.nan_to_num(probs, nan=0.0)
    total = probs.sum()
    if total == 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / total
    return np.random.choice(len(probs), p=probs)


def selfplay_worker(worker_id, strong_model_state, weak_model_state, data_queue, stop_event, num_simulations,
                    board_size, opponent_type="weak_mcts", opponent_simulations=None):
    device = torch.device('cpu')
    torch.set_num_threads(min(mp.cpu_count() // 4, 4))

    local_strong = PolicyValueNet(board_size=board_size).to(device)
    local_strong.load_state_dict(strong_model_state)

    local_weak = None
    if weak_model_state:
        local_weak = PolicyValueNet(board_size=board_size).to(device)
        local_weak.load_state_dict(weak_model_state)

    opponent_simulations = opponent_simulations or max(1, num_simulations // 10)

    while not stop_event.is_set():
        board = GomokuBoard(size=board_size)
        player = 1
        strong_mcts = MCTS(local_strong)
        weak_mcts = MCTS(local_weak) if local_weak else None

        while not board.is_terminal() and not stop_event.is_set():
            if player == 1:
                _, probs = strong_mcts.run(board, player, number_samples=num_simulations, is_train=True)
            else:
                if opponent_type == "random":
                    move = random.choice(board.legal_moves())
                    board.step(move, player)
                    player = -player
                    continue
                else:
                    _, probs = weak_mcts.run(board, player, number_samples=opponent_simulations, is_train=False)

            legal = board.legal_moves()
            flat_probs = probs.reshape(-1)
            mask = np.zeros_like(flat_probs, dtype=bool)
            for yx in legal:
                idx = yx[0] * board.size + yx[1]
                mask[idx] = True
            flat_probs = np.where(mask, flat_probs, 0.0)
            s = flat_probs.sum()
            if s > 0:
                flat_probs /= s
            else:
                flat_probs[mask] = 1.0 / len(legal)

            y, x = divmod(safe_choice(flat_probs), board.size)
            board.step((y, x), player)
            player = -player

        boards, policies, values, _ = strong_mcts.get_train_data(game_result=board.winner())
        for b, p, v in zip(boards, policies, values):
            while not stop_event.is_set():
                try:
                    data_queue.put((b, p.reshape(-1), v), timeout=1)
                    break
                except mp.queues.Full:
                    continue


def train_realtime(args, update_threshold=0.6):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs('./check_dir', exist_ok=True)
    run_id = len(os.listdir('./check_dir'))
    checkpoints_path = f"./check_dir/run{run_id}"
    os.makedirs(checkpoints_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    strong_model = PolicyValueNet(board_size=args.board_size).to(device)
    weak_model = PolicyValueNet(board_size=args.board_size).to(device)
    # 定义优化器
    optimizer = torch.optim.Adam(strong_model.parameters(), lr=0.2)

    milestones = [30, 60, 90]
    gamma = 0.1  # 每次降低到原来的 0.1
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    writer = SummaryWriter(os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S")))

    data_queue = mp.Queue(maxsize=args.queue_size)
    stop_event = mp.Event()

    strong_model_state = {k: v.cpu() for k, v in strong_model.state_dict().items()}
    weak_model_state = {k: v.cpu() for k, v in weak_model.state_dict().items()}

    workers = []
    for i in range(args.num_workers):
        p = mp.Process(target=selfplay_worker, args=(
            i, strong_model_state, weak_model_state, data_queue, stop_event,
            args.num_simulations, args.board_size, args.opponent_type, args.opponent_simulations
        ))
        p.start()
        workers.append(p)

    global_step = 0
    recent_results = []
    last_print_time = time.time()

    try:
        for step in tqdm.trange(args.train_steps, desc="训练中"):
            batch = []
            while len(batch) < args.batch_size:
                try:
                    batch.append(data_queue.get(timeout=5))
                    current_time = time.time()
                    if current_time - last_print_time >= 300:
                        print(f"[训练步骤 {step}] 当前 batch 长度: {len(batch)}")
                        last_print_time = current_time
                except Exception:
                    if stop_event.is_set():
                        break

            if len(batch) == 0:
                continue

            # 确保队列中剩余数据也被利用
            while not data_queue.empty() and len(batch) < args.batch_size:
                try:
                    batch.append(data_queue.get_nowait())
                except Exception:
                    break

            boards = torch.stack([b for b, _, _ in batch]).to(device)
            policies = torch.stack([p for _, p, _ in batch]).to(device)
            values = torch.tensor([v for _, _, v in batch], dtype=torch.float32).reshape(-1, 1).to(device)

            pred_pi, pred_v = strong_model(boards)
            policy_loss = -(policies * torch.log(pred_pi + 1e-8)).sum(dim=1).mean()
            value_loss = F.mse_loss(pred_v, values)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # TensorBoard 记录
            writer.add_scalar('loss/total_loss', loss.item(), global_step)
            writer.add_scalar('loss/policy_loss', policy_loss.item(), global_step)
            writer.add_scalar('loss/value_loss', value_loss.item(), global_step)
            writer.add_scalar('metric/avg_winner_rate', float(np.mean(recent_results)) if recent_results else 0.0,
                              global_step)
            writer.add_scalar('lr/current_lr', scheduler.get_last_lr()[0], global_step)
            winner_rate = (values > 0).float().mean().item()
            recent_results.append(winner_rate)
            if len(recent_results) > 100:
                recent_results.pop(0)

            if np.mean(recent_results) >= update_threshold:
                weak_model.load_state_dict(strong_model.state_dict())

            if step % args.save_interval == 0:
                torch.save(strong_model.state_dict(), os.path.join(checkpoints_path, f"model_step{step}.pth"))
                print(f"[保存模型] 第 {step} 步模型已保存")

            global_step += 1

    finally:
        stop_event.set()
        for p in workers:
            p.join()
        torch.save(strong_model.state_dict(), args.save_path)
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='实时训练：强模型 vs 弱模型自对弈 MCTS')
    parser.add_argument('--board-size', type=int, default=15)
    parser.add_argument('--num-workers', type=int, default=18)
    parser.add_argument('--num-simulations', type=int, default=800)
    parser.add_argument('--opponent-type', type=str, choices=['random', 'weak_mcts'], default='weak_mcts')
    parser.add_argument('--opponent-simulations', type=int, default=100)
    parser.add_argument('--train-steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--save-interval', type=int, default=20)
    parser.add_argument('--queue-size', type=int, default=512)
    parser.add_argument('--log-dir', type=str, default='./tf-logs/')
    parser.add_argument('--save-path', type=str, default='realtime_model.pth')
    parser.add_argument('--no-cuda', action='store_true', help='禁用 CUDA')
    parser.add_argument('--update-threshold', type=float, default=0.6, help='弱模型更新胜率阈值')
    args = parser.parse_args()

    print(f"[训练开始] 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    train_realtime(args, update_threshold=args.update_threshold)
    print(f"[训练结束] 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
