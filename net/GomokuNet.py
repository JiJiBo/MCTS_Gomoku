import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        return out + x


class PolicyValueNet(nn.Module):
    """
    输入: x[B, C, H, W]
    输出: policy_logits[B, H*W], value[B, 1]
    """

    def __init__(self, in_channels=4, channels=128, num_blocks=8, board_size=15):
        super().__init__()
        self.H = self.W = board_size
        # Stem
        self.stem = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        # Residual trunk
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        # Policy head
        self.policy_bn = nn.BatchNorm2d(channels)
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)  # 2通道足够
        self.policy_fc = nn.Linear(2 * self.H * self.W, self.H * self.W)
        # Value head
        self.value_bn = nn.BatchNorm2d(channels)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_fc1 = nn.Linear(self.H * self.W, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # 初始化（Kaiming）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias) if m.bias is not None else None

    def forward(self, x):
        # x: [B,C,H,W]
        out = F.relu(self.stem_bn(self.stem(x)))
        out = self.blocks(out)

        # Policy head
        p = F.relu(self.policy_bn(out))
        p = self.policy_conv(p)  # [B,2,H,W]
        p = p.view(p.size(0), -1)  # [B, 2*H*W]
        policy_logits = self.policy_fc(p)  # [B, H*W]

        # Value head
        v = F.relu(self.value_bn(out))
        v = self.value_conv(v)  # [B,1,H,W]
        v = v.view(v.size(0), -1)  # [B, H*W]
        v = F.relu(self.value_fc1(v))  # [B,256]
        value = torch.tanh(self.value_fc2(v))  # [B,1]
        policy_logits = F.softmax(policy_logits, dim=1)
        return policy_logits, value

    def calc_board(self, board_4ch):
        with torch.no_grad():
            policy_logits, value = self.forward(board_4ch)
            policy_logits = policy_logits.view(policy_logits.size(0), self.H, self.W)
            value = value.view(value.size(0), 1)
            policy = policy_logits.cpu().detach().numpy().tolist()
            value = value.cpu().detach().numpy().tolist()
        return policy, value

    def calc_one_board(self, board_4ch):
        if isinstance(board_4ch, np.ndarray):
            board_4ch = torch.from_numpy(board_4ch)
        device = next(self.parameters()).device
        board_4ch = board_4ch.unsqueeze(0).to(device, dtype=torch.float32)
        with torch.no_grad():
            policy_logits, value = self.forward(board_4ch)
        policy_logits = (
            policy_logits.squeeze(0).view(self.H, self.W).detach().cpu().numpy()
        )
        value = float(value.squeeze().item())
        return policy_logits, value
