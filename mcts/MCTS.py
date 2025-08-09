import math
import random

import numpy as np
import torch
from jinja2.nodes import Continue
from networkx.classes import edges

from core.board import GomokuBoard
from mcts.MCTS_Node import MCTSNode, Edge
from net.GomokuNet import PolicyValueNet


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
        probs = np.zeros((root_node.board.size, root_node.board.size))
        total_visits = sum(
            edge.child.visit_count if edge.child is not None else 0 for edge in root_node.children.values())
        if not is_train:
            for move, edg, in root_node.children.items():
                if edg.child is not None:
                    i, j = move
                    probs[i][j] = edg.child.visit_count / total_visits if total_visits > 0 else 0
            return root_node.q_value(), probs
        else:
            policy_logits, value = self.model.calc_one_board(root_node.board)
            good = len(root_node.board.legal_moves())
            for i in range(root_node.board.size):
                for j in range(root_node.board.size):
                    if root_node.board.board[j][i] != 0 or policy_logits[j][i] == 0:
                        probs[j][i] = 0
                    else:
                        probs[j][i] = policy_logits[j][i] / good
            sum_used = 0
            moves = []
            for move, edg in root_node.children.items():
                if edg.child is not None and edg.child.visit_count != 0:
                    sum_used += probs[move]
                    probs[move] = 0
                    moves.append((move[0], move[1], edg.child))
            moves.sort(key=lambda x: x[0], reverse=True)
            value_sum = 0
            for i in range(0, len(moves)):
                cnt = moves[i][2].visit_count
                if i + 1 < len(moves):
                    cnt += moves[i + 1][2].visit_count
                if cnt == 0:
                    continue
                cur = 1e-9
                best_pos = i
                for j in range(0, i + 1):
                    vals = -moves[j][2].q_value()
                    if vals > cur:
                        cur = vals
                        best_pos = j
                probs[moves[best_pos][0], moves[best_pos][1]] += sum_used * cnt * (i + 1) / total_visits
                value_sum += cur * cnt(i + 1) / total_visits
            for i in range(root_node.board.size):
                for j in range(root_node.board.size):
                    if probs[j][i] < 0:
                        print(good)
                        print(probs[j][i])
                        assert False
            return value_sum, probs

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
