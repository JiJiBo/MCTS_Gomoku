import copy

import numpy as np


class GomokuBoard:
    def __init__(self, size=15, count_win=5):
        self.size = size
        self.count_win = count_win
        self.board = np.zeros((size, size), dtype=np.int8)  # -1, 0, +1
        self.move_count = 0
        self.last_move = None
        self.history = []  # [(y,x,player_flag)]
        self.win_path = None  # 连成五子的坐标序列

    def reset(self):
        self.board.fill(0)
        self.move_count = 0
        self.last_move = None
        self.history.clear()
        self.win_path = None

    def legal_moves(self):
        ys, xs = np.where(self.board == 0)
        return list(zip(ys.tolist(), xs.tolist()))

    def legal_mask(self, flat=True):
        m = (self.board == 0).astype(np.float32)
        return m.reshape(-1) if flat else m

    def step(self, move, player_flag, return_info: bool = False):
        """默认兼容旧返回；若 return_info=True，返回 (board, winner, done)"""
        y, x = move
        if player_flag not in (-1, 1):
            raise ValueError("player_flag must be +1 (Black) or -1 (White).")
        if not (0 <= y < self.size and 0 <= x < self.size):
            raise ValueError("move out of board.")
        if self.board[y, x] != 0:
            raise ValueError("illegal move: cell occupied.")

        self.board[y, x] = player_flag
        self.move_count += 1
        self.last_move = (y, x)
        self.history.append((y, x, player_flag))

        w = self.winner_from_last()
        done = (w != 0) or (self.move_count == self.size * self.size)

        if return_info:
            return self.board, w, done
        else:
            # 兼容你之前的用法
            return self.board, player_flag

    def undo(self):
        """回退一步；若无步可退则抛错"""
        if not self.history:
            raise RuntimeError("no move to undo")
        y, x, _ = self.history.pop()
        self.board[y, x] = 0
        self.move_count -= 1
        self.last_move = (self.history[-1][0], self.history[-1][1]) if self.history else None
        self.win_path = None

    def is_terminal(self):
        return self.winner_from_last() != 0 or self.move_count == self.size * self.size

    def play_count(self):
        return self.move_count

    def winner(self):
        """1(黑胜)/-1(白胜)/0(未分胜负或和棋)"""
        if self.move_count == self.size * self.size and self.winner_from_last() == 0:
            return 0  # 和棋 or 未分胜负（交由外部 is_terminal 判断）
        return self.winner_from_last()

    # ---- 仅检查上一手，O(K) ----
    def winner_from_last(self):
        if self.last_move is None:
            self.win_path = None
            return 0
        y0, x0 = self.last_move
        p = self.board[y0, x0]
        if p == 0:
            self.win_path = None
            return 0
        K = self.count_win
        s = self.size
        b = self.board
        for dy, dx in ((0, 1), (1, 0), (1, 1), (-1, 1)):
            seq = [(y0, x0)]
            y, x = y0 - dy, x0 - dx
            while 0 <= y < s and 0 <= x < s and b[y, x] == p:
                seq.insert(0, (y, x))
                y -= dy
                x -= dx
            y, x = y0 + dy, x0 + dx
            while 0 <= y < s and 0 <= x < s and b[y, x] == p:
                seq.append((y, x))
                y += dy
                x += dx
            if len(seq) >= K:
                self.win_path = seq[:K]
                return int(p)
        self.win_path = None
        return 0

    # ---- 4通道特征 ----
    def get_planes_4ch(self, current_player: int):
        """
        [4,H,W]:
          0 我方(相对 current_player) 1/0
          1 对方 1/0
          2 空白 1/0
          3 上一步 one-hot
        """
        if current_player not in (-1, 1):
            raise ValueError("current_player must be +1 or -1")
        b = self.board
        me = (b == current_player).astype(np.float32)
        opp = (b == -current_player).astype(np.float32)
        empty = (b == 0).astype(np.float32)
        last = np.zeros_like(b, dtype=np.float32)
        if self.last_move is not None:
            y, x = self.last_move
            last[y, x] = 1.0
        return np.stack([me, opp, empty, last], axis=0).astype(np.float32)

    def copy(self):
        new_board = GomokuBoard(self.size, self.count_win)
        new_board.board = copy.deepcopy(self.board)
        new_board.move_count = self.move_count
        return new_board

    def is_win(self, player_flag):
        """检查某个玩家是否获胜"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y, x] != player_flag:
                    continue
                for dy, dx in directions:
                    count = 1
                    ny, nx = y + dy, x + dx
                    while 0 <= ny < self.size and 0 <= nx < self.size and self.board[ny, nx] == player_flag:
                        count += 1
                        if count >= self.count_win:
                            return True
                        ny += dy
                        nx += dx
        return False

    def is_full(self):
        return self.move_count >= self.size * self.size

    def evaluation(self):
        board_size = self.size
        # 已下子数，用于早赢奖励（和你原逻辑一致）
        num_used = int((self.board != 0).sum())

        # 四个方向：水平、垂直、主对角、反对角
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for y in range(board_size):
            for x in range(board_size):
                p = self.board[y][x]
                if p == 0:
                    continue
                for dy, dx in directions:
                    cnt = 0
                    for d in range(5):
                        ny = y + d * dy
                        nx = x + d * dx
                        if 0 <= ny < board_size and 0 <= nx < board_size and self.board[ny][nx] == p:
                            cnt += 1
                        else:
                            break
                    if cnt == 5:
                        score = (1 - num_used * 3e-4)
                        return score if p == 1 else -score
        return 0
