import argparse
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from core.board import GomokuBoard
from mcts.MCTS import MCTS
from net.GomokuNet import PolicyValueNet


def generate_self_play_data(model, games, board_size, simulations):
    mcts = MCTS(model)
    boards, policies, values = [], [], []
    for _ in range(games):
        board = GomokuBoard(board_size)
        player = 1
        while not board.is_terminal():
            _, probs = mcts.run(board, number_samples=simulations, is_train=True)
            moves = board.legal_moves()
            move_probs = np.array([probs[y, x] for y, x in moves], dtype=np.float32)
            if move_probs.sum() <= 0:
                move_probs = np.ones(len(moves), dtype=np.float32) / len(moves)
            else:
                move_probs /= move_probs.sum()
            move = moves[np.random.choice(len(moves), p=move_probs)]
            board.step(move, player)
            player = -player
        b, p, v, _ = mcts.get_train_data()
        boards.extend(b)
        policies.extend(p)
        values.extend(v)
    boards = torch.stack(boards)
    policies = torch.stack([pi.view(-1) for pi in policies])
    values = torch.tensor(values, dtype=torch.float32)
    return TensorDataset(boards, policies, values)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    model = PolicyValueNet(board_size=args.board_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dataset = generate_self_play_data(model, args.self_play_games, args.board_size, args.simulations)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    writer = SummaryWriter(args.log_dir)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for boards, target_pi, target_v in loader:
            boards = boards.to(device)
            target_pi = target_pi.to(device)
            target_v = target_v.to(device).unsqueeze(-1)

            pred_pi, pred_v = model(boards)
            policy_loss = -(target_pi * torch.log(pred_pi + 1e-8)).sum(dim=1).mean()
            value_loss = F.mse_loss(pred_v, target_v)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss/policy', policy_loss.item(), global_step)
            writer.add_scalar('loss/value', value_loss.item(), global_step)
            writer.add_scalar('loss/total', loss.item(), global_step)
            global_step += 1

        print(f'Epoch {epoch}/{args.epochs} - loss: {loss.item():.4f}')

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Gomoku policy-value network with self-play data')
    parser.add_argument('--self-play-games', type=int, default=10, help='number of self-play games to generate')
    parser.add_argument('--simulations', type=int, default=100, help='MCTS simulations per move')
    parser.add_argument('--save-path', type=str, default='policy_value_net.pth', help='path to save the trained model')
    parser.add_argument('--log-dir', type=str, default='runs', help='directory for TensorBoard logs')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--board-size', type=int, default=15)
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA')
    args = parser.parse_args()

    train(args)
