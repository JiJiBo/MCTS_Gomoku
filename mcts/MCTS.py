import math
import random
import numpy as np
import torch
import logging  # 导入日志模块

from core.board import GomokuBoard
from mcts.MCTS_Node import MCTSNode, Edge
from net.GomokuNet import PolicyValueNet

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class MCTS():
    def __init__(self, model: PolicyValueNet, use_rand=0.1, c_puct=1.4):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.use_rand = use_rand
        self.c_puct = c_puct
        self.visit_nodes = []

    def run(self, root_board: GomokuBoard, player: int, number_samples=100, is_train=False):
        """Run MCTS search from the given board state."""
        root_node = MCTSNode(root_board, player=player)
        self.visit_nodes.append(root_node)
        for _ in range(number_samples):
            node = root_node
            search_path = [node]
            while node.children:
                node = self.select_child(node)
                search_path.append(node)
            if node.board.is_terminal():
                value = node.board.evaluation()
            else:
                value = self.expand_node(node)
            for n in reversed(search_path):
                n.visit_count += 1
                n.total_value += value
                value = -value
        return self.get_result(root_node, is_train)

    def get_result(self, root_node: MCTSNode, is_train):
        """Get the result of the MCTS search."""
        size = root_node.board.size
        board_mat = root_node.board.board
        legal_mask = (board_mat == 0)
        probs = np.zeros((size, size), dtype=np.float32)

        items = []  # (move, child, prior)
        for move, edge in root_node.children.items():
            items.append((move, edge.child, edge.prior))

        visit_counts = np.array(
            [(c.visit_count if c is not None else 0) for (_, c, _) in items],
            dtype=np.float32
        ) if items else np.empty((0,), dtype=np.float32)

        total_visits = float(visit_counts.sum()) if visit_counts.size > 0 else 0.0

        if getattr(root_node, "visit_count", 0) > 0:
            root_value = float(root_node.q_value())
        else:
            policy_logits_net, value_net = self.model.calc_one_board(
                root_node.board.get_planes_9ch(root_node.player)
            )
            root_value = float(value_net)

        def assign_by_moves(moves_list, weights):
            for (mv, _child, _prior), w in zip(moves_list, weights):
                y, x = mv
                probs[y, x] = float(w)

        if not is_train:
            temperature_eval = 1e-3
            if total_visits > 0:
                if temperature_eval <= 0:
                    idx = int(np.argmax(visit_counts))
                    y, x = items[idx][0]
                    probs[y, x] = 1.0
                else:
                    x = visit_counts ** (1.0 / max(1e-6, temperature_eval))
                    Z = float(x.sum())
                    if Z > 0:
                        assign_by_moves(items, x / Z)
            else:
                policy_logits_net, _ = self.model.calc_one_board(
                    root_node.board.get_planes_9ch(root_node.player)
                )
                prior = np.where(legal_mask, policy_logits_net, 0.0).astype(np.float32)
                s = float(prior.sum())
                if s > 0:
                    probs = prior / s
                else:
                    cnt = int(legal_mask.sum())
                    if cnt > 0:
                        probs = legal_mask.astype(np.float32) / float(cnt)
            return root_value, probs

        temperature_train = 1.0
        prior_backfill_eps = 0.10

        policy_logits_net, _ = self.model.calc_one_board(
            root_node.board.get_planes_9ch(root_node.player)
        )
        prior = np.where(legal_mask, policy_logits_net, 0.0).astype(np.float32)
        ps = float(prior.sum())
        if ps > 0:
            prior /= ps
        else:
            cnt = int(legal_mask.sum())
            if cnt > 0:
                prior = legal_mask.astype(np.float32) / float(cnt)

        if total_visits > 0:
            if temperature_train <= 0:
                idx = int(np.argmax(visit_counts))
                y, x = items[idx][0]
                probs[y, x] = 1.0
            else:
                x = visit_counts ** (1.0 / max(1e-6, temperature_train))
                Z = float(x.sum())
                if Z > 0:
                    assign_by_moves(items, x / Z)

            if prior_backfill_eps > 0.0:
                probs *= (1.0 - prior_backfill_eps)

                visited_mask = np.zeros_like(legal_mask, dtype=bool)
                for (mv, child, _), _v in zip(items, visit_counts):
                    if child is not None and child.visit_count > 0:
                        vy, vx = mv
                        visited_mask[vy, vx] = True

                unvisited_mask = legal_mask & (~visited_mask)
                prior_unvisited = np.where(unvisited_mask, prior, 0.0)
                su = float(prior_unvisited.sum())
                if su > 0:
                    probs += prior_backfill_eps * (prior_unvisited / su)
                else:
                    s2 = float(probs.sum())
                    if s2 > 0:
                        probs /= s2
        else:
            probs = prior.copy()

        probs = np.where(legal_mask, probs, 0.0).astype(np.float32)
        Z = float(probs.sum())
        if Z > 0:
            probs /= Z
        else:
            cnt = int(legal_mask.sum())
            if cnt > 0:
                probs = legal_mask.astype(np.float32) / float(cnt)

        return root_value, probs

    def select_child(self, node: MCTSNode):
        total_visits = node.visit_count
        explore_term = math.sqrt(total_visits + 1)
        best_score = -float("inf")
        best_move = None
        for move, edge in node.children.items():
            child, prior = edge.child, edge.prior
            v_count = child.visit_count if child is not None else 0
            q = child.q_value() if v_count > 0 else 0.0
            u = self.c_puct * prior * explore_term / (1 + v_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move

        edge = node.children[best_move]
        child, prior = edge.child, edge.prior
        if child is None:
            y, x = best_move
            new_board = node.board.copy()
            new_board.step((y, x), node.player)
            child = MCTSNode(new_board, parent=node, move=best_move, player=-node.player)
            node.children[best_move] = Edge(child, prior)
        return child

    def expand_node(self, node: MCTSNode):
        policy_logits, value = self.model.calc_one_board(
            torch.from_numpy(node.board.get_planes_9ch(node.player))
        )
        moves = node.board.legal_moves()
        priors = np.array([policy_logits[y, x] for y, x in moves], dtype=np.float32)
        s = float(priors.sum())
        if s > 0:
            priors /= s
        else:
            priors = np.ones(len(moves), dtype=np.float32) / max(1, len(moves))
        priors = [max(0.0, p + random.normalvariate(0, self.use_rand)) for p in priors]
        ps = float(sum(priors))
        priors = [p / ps if ps > 0 else 1.0 / len(priors) for p in priors]
        for (y, x), p in zip(moves, priors):
            node.children[(y, x)] = Edge(None, float(p))

        return float(value)

    def get_train_data(self, game_result: int | None = None):
        """Collect training samples from all visited root nodes."""
        logging.info(f"Collecting training data with game result {game_result}.")
        boards, policies, values, weights = [], [], [], []

        for root in self.visit_nodes:
            size = root.board.size
            probs = np.zeros((size, size), dtype=np.float32)

            total_visits = 0
            for move, edge in root.children.items():
                child = edge.child
                if child is not None and child.visit_count > 0:
                    y, x = move
                    probs[y, x] = child.visit_count
                    total_visits += child.visit_count

            if total_visits == 0:
                logging.warning(f"No valid visits for root node {root}. Skipping.")
                continue

            probs /= float(total_visits)

            board_tensor = torch.from_numpy(
                root.board.get_planes_9ch(root.player)
            ).float()

            boards.append(board_tensor)
            policies.append(torch.from_numpy(probs).float())
            if game_result is None:
                values.append(float(root.q_value()))
            else:
                values.append(float(game_result if root.player == 1 else -game_result))
            weights.append(math.sqrt(total_visits))

        self.visit_nodes.clear()

        logging.info(f"Collected {len(boards)} training samples.")
        return boards, policies, values, weights
