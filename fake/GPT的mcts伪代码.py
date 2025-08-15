import math
import random


class Node:
    def __init__(self, state, parent=None, prior=0):
        self.state = state  # 当前棋局状态
        self.parent = parent  # 父节点
        self.children = {}  # 子节点字典: action -> Node
        self.P = prior  # 神经网络预测概率
        self.N = 0  # 当前动作访问次数
        self.W = 0  # 当前动作累计胜利值
        self.Q = 0  # 当前动作平均胜利值


# PUCT选择动作函数
def select_action(node, c_puct=1.0):
    best_score = -float('inf')
    best_action = None
    for a, child in node.children.items():
        Q = child.Q
        P = child.P
        N_a = child.N
        N_parent = node.N
        score = Q + c_puct * P * math.sqrt(N_parent) / (1 + N_a)
        if score > best_score:
            best_score = score
            best_action = a
    return best_action


# MCTS搜索函数
def mcts_search(root, neural_network, num_simulations=800, c_puct=1.0):
    for _ in range(num_simulations):
        node = root
        path = [node]

        # 1. 选择节点
        while node.children:
            action = select_action(node, c_puct)
            node = node.children[action]
            path.append(node)

        # 2. 扩展节点
        if not node.state.is_terminal():
            policy, value = neural_network.predict(node.state)
            for action, p in policy.items():
                if action not in node.children:
                    node.children[action] = Node(node.state.next_state(action), parent=node, prior=p)
        else:
            value = node.state.get_reward()  # 终局价值

        # 3. 回溯更新
        for n in reversed(path):
            n.N += 1
            n.W += value
            n.Q = n.W / n.N
            value = -value  # 交替玩家

# 使用示例
# root = Node(initial_board_state)
# mcts_search(root, neural_network, num_simulations=800)
