import argparse
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


def selfplay_worker(worker_id: int, model: PolicyValueNet, data_queue: queue.Queue,
                    stop_event: threading.Event, model_lock: threading.Lock,
                    num_simulations: int):
    """Generate self-play data using MCTS and push to a queue."""
    print(f"[Worker {worker_id}] Starting self-play worker.")
    local_model = PolicyValueNet(board_size=model.H)
    while not stop_event.is_set():
        # Refresh local model parameters
        with model_lock:
            local_model.load_state_dict(model.state_dict())
        mcts = MCTS(local_model)
        board = GomokuBoard(size=model.H)
        player = 1
        while not board.is_terminal() and not stop_event.is_set():
            _, probs = mcts.run(board, player, number_samples=num_simulations, is_train=True)
            flat = probs.reshape(-1)
            move = np.random.choice(len(flat), p=flat)
            y, x = divmod(move, board.size)
            board.step((y, x), player)
            player = -player
        result = board.winner()
        boards, policies, values, _ = mcts.get_train_data(game_result=result)
        for b, p, v in zip(boards, policies, values):
            data_queue.put((b, p.view(-1), v))
    print(f"[Worker {worker_id}] Stopped self-play worker.")


def train_realtime(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model = PolicyValueNet(board_size=args.board_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(args.log_dir)

    data_queue: queue.Queue = queue.Queue(maxsize=args.queue_size)
    stop_event = threading.Event()
    model_lock = threading.Lock()

    workers = []
    for i in range(args.num_workers):
        t = threading.Thread(target=selfplay_worker,
                             args=(i, model, data_queue, stop_event, model_lock, args.num_simulations),
                             daemon=True)
        t.start()
        workers.append(t)

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

            # 打印训练信息
            print(
                f"[Train Step {step}] Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Total Loss: {loss.item():.4f}")
            global_step += 1

    finally:
        stop_event.set()
        for t in workers:
            t.join()
        torch.save(model.state_dict(), args.save_path)
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time training with self-play MCTS data')
    parser.add_argument('--board-size', type=int, default=15)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--num-simulations', type=int, default=100)
    parser.add_argument('--train-steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--queue-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log-dir', type=str, default='runs/realtime')
    parser.add_argument('--save-path', type=str, default='realtime_model.pth')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA')
    args = parser.parse_args()

    # 开始训练前输出开始时间
    print(f"[Training Start] Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    train_realtime(args)
    # 训练完成时输出结束时间
    print(f"[Training End] Training ended at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
