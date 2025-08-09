import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import GomokuDataset
from net.GomokuNet import PolicyValueNet


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    dataset = GomokuDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = PolicyValueNet(board_size=args.board_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
    parser = argparse.ArgumentParser(description='Train Gomoku policy-value network')
    parser.add_argument('--data-dir', type=str, required=True, help='directory containing training .npz files')
    parser.add_argument('--save-path', type=str, default='policy_value_net.pth', help='path to save the trained model')
    parser.add_argument('--log-dir', type=str, default='runs', help='directory for TensorBoard logs')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--board-size', type=int, default=15)
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA')
    args = parser.parse_args()

    train(args)
