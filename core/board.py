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
            return self.board, player_flag

    def undo(self):
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
        if self.move_count == self.size * self.size and self.winner_from_last() == 0:
            return 0
        return self.winner_from_last()

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

    def get_planes_4ch(self, current_player: int):
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

    def get_planes_9ch(self, current_player: int, history_len: int = 4):
        """
        [9,H,W]:
          - 前4步的历史局面，每步2层(黑,白)
          - 当前走子方指示平面(全1/0)
        """
        if current_player not in (-1, 1):
            raise ValueError("current_player must be +1 or -1")

        H, W = self.size, self.size
        planes = []
        # 最近 history_len 步（不足则补零）
        for i in range(history_len):
            idx = -(i + 1)
            if len(self.history) + idx < 0:
                planes.append(np.zeros((H, W), dtype=np.float32))
                planes.append(np.zeros((H, W), dtype=np.float32))
            else:
                board_copy = np.zeros((H, W), dtype=np.int8)
                for (y, x, p) in self.history[: idx + 1]:
                    board_copy[y, x] = p
                me = (board_copy == current_player).astype(np.float32)
                opp = (board_copy == -current_player).astype(np.float32)
                planes.append(me)
                planes.append(opp)

        # 当前走子方指示通道
        indicator = np.full((H, W), 1.0 if current_player == 1 else 0.0, dtype=np.float32)
        planes.append(indicator)
        return np.stack(planes, axis=0).astype(np.float32)

    def copy(self):
        new_board = GomokuBoard(self.size, self.count_win)
        new_board.board = copy.deepcopy(self.board)
        new_board.move_count = self.move_count
        return new_board

    def is_win(self, player_flag):
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
