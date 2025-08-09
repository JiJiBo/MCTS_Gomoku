from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from core.board import GomokuBoard

Move = Tuple[int, int]


@dataclass
class Edge:
    child: Optional['MCTSNode']
    prior: float


class MCTSNode:
    def __init__(self, board: GomokuBoard, player: int, move=None, parent=None, prior_prob=1.0):
        self.board = board  # 当前棋盘
        self.player = player  # 当前执棋方
        self.move = move  # 父节点走到此节点的着法
        self.parent = parent
        self.children: Dict[Move, Edge] = {}  # move -> MCTSNode

        # MCTS 统计
        self.visit_count = 0  # N
        self.total_value = 0.0  # W
        self.prior_prob = prior_prob  # P

        # 还没探索过的走法（合法空位）
        self.untried_moves = board.legal_moves()

        # 是否终局
        self.is_terminal = (
                board.is_win(1) or board.is_win(-1) or board.is_full()
        )

    def q_value(self):
        """平均价值"""
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

    def u_value(self, c_puct=1.4):
        """PUCT 中的探索项"""
        if self.parent is None:
            return 0
        return c_puct * self.prior_prob * (
                (self.parent.visit_count ** 0.5) / (1 + self.visit_count)
        )
