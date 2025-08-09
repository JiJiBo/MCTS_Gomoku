import math
import random

import torch

from  core.board import GomokuBoard
from  mcts.MCTS_Node import MCTSNode, Edge
from  net.GomokuNet import PolicyValueNet


class MCTS():
    def __init__(self, model: PolicyValueNet, use_rand=0.1, c_puct=1.4):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.use_rand = use_rand
        self.c_puct = c_puct
        self.visit_nodes = []

    def run(self, root_board: GomokuBoard, number_samples=100, is_train=False):
        root_node = MCTSNode(root_board, player=1)
        self.visit_nodes.append(root_node)
        for si in range(number_samples):
            node = root_node
            search_path = [node]
            while node.children:
                node = self.select_child(node)
                search_path.append(node)
            if not node.board.is_terminal():
                self.expand_node(node)
            else:
                node.prior_prob = node.board.evaluation()
            prior_prob = node.board.evaluation()
            for node in reversed(search_path):
                node.visit_count += 1
                node.prior_prob = prior_prob
                prior_prob = -prior_prob
        return self.get_result(root_node, is_train)

    def get_result(self, root_node: MCTSNode, is_train):
        pass

    def select_child(self, node: MCTSNode):
        total_visits = sum(
            (edge.child.visit_count if edge.child is not None else 0)
            for edge in node.children.values()
        )

        explore_buff = math.pow(total_visits + 1, 0.5)
        best_score = 0
        best_move = None
        # 遍历每一个子节点
        # 计算 Q 和 U
        for move, edge in node.children.items():
            child, prior = edge.child, edge.prior
            if child is not None and child.visit_count > 0:
                # Q 平均价值：从状态 s 走 a 后的平均胜率（累计价值 / 访问次数）
                # Q 利用
                Q = child.q_value()
                # U  探索
                U = child.u_value(self.c_puct)
            else:
                # Q 平均价值：从状态 s 走 a 后的平均胜率（累计价值 / 访问次数）
                # Q 利用
                Q = prior / child.visit_count if child is not None else 0
                # U  探索
                U = self.c_puct * prior * explore_buff / (1 + child.visit_count if child is not None else 0)
            score = Q + U
            if score > best_score:
                best_score = score
                best_move = move
        edge = node.children[best_move]
        child, prior = edge.child, edge.prior
        if child is None:
            y, x = best_move
            new_board = node.board.copy()
            new_board[y, x] = node.player
            for y in range(node.board.size):
                for x in range(node.board.size):
                    new_board[y][x] *= -1
            child = MCTSNode(new_board, parent=node, move=best_move, player=-node.player)
            node.children[best_move] = Edge(child, prior)
        return child

    def expand_node(self, node: MCTSNode):
        policy_logits, value = self.model.calc_one_board(node.board.get_planes_4ch(node.player))
        can_taps = node.board.legal_moves()
        sum_1 = 0
        for can_tap in can_taps:
            y, x = can_tap
            p = policy_logits[y, x]
            sum_1 += p
        if sum_1 == 0:
            sum_1 = 1e-9
        for can_tap in can_taps:
            y, x = can_tap
            if node.board.board[y][x] == 0:
                prior = float(policy_logits / sum_1 + random.normalvariate(mu=0, sigma=self.use_rand))
                node.children[(y, x)] = Edge(None, prior)
