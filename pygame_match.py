import sys
import time
import random
import numpy as np
import pygame
import torch
from typing import Optional

from core.board import GomokuBoard
from mcts.MCTS import MCTS
from net.GomokuNet import PolicyValueNet


class Agent:
    """Base agent interface."""

    def select_move(self, board: GomokuBoard, player: int):
        raise NotImplementedError


class RandomAgent(Agent):
    """Agent that plays random legal moves."""

    def select_move(self, board: GomokuBoard, player: int):
        moves = board.legal_moves()
        return random.choice(moves)


class MCTSAgent(Agent):
    """Agent that uses MCTS with a neural network model."""

    def __init__(self, model: PolicyValueNet, simulations: int = 100):
        self.mcts = MCTS(model)
        self.simulations = simulations

    def select_move(self, board: GomokuBoard, player: int):
        _, probs = self.mcts.run(board, player, self.simulations)
        idx = int(np.argmax(probs))
        y, x = divmod(idx, board.size)
        return (y, x)


class ModelAgent(MCTSAgent):
    """Agent that loads a PolicyValueNet model from a checkpoint."""

    def __init__(
            self,
            model_path: str,
            board_size: int = 15,
            device: Optional[str] = None,
            simulations: int = 100,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PolicyValueNet(board_size=board_size)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        super().__init__(model, simulations)


class PygameMatch:
    """Display a real-time match between two agents or a human using pygame."""

    def __init__(self, agent_black: Optional[Agent], agent_white: Optional[Agent],
                 board_size: int = 15, cell_size: int = 40, margin: int = 20,
                 delay: int = 500):
        self.board = GomokuBoard(board_size)
        self.agent_black = agent_black
        self.agent_white = agent_white
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = margin
        self.delay = delay

        pygame.init()
        size = margin * 2 + cell_size * (board_size - 1)
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("Gomoku Match")

    # ---- helpers ----
    def is_human_turn(self, player: int) -> bool:
        return (player == 1 and self.agent_black is None) or (
                player == -1 and self.agent_white is None
        )

    def apply_undo(self) -> int:
        steps = 2 if ((self.agent_black is None) ^ (self.agent_white is None)) else 1
        for _ in range(min(steps, len(self.board.history))):
            self.board.undo()
        # 更新赢家轨迹
        self.board.winner_from_last()
        return 1 if self.board.move_count % 2 == 0 else -1

    def get_human_move(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_u:
                    return 'undo'
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    x = int(round((mx - self.margin) / self.cell_size))
                    y = int(round((my - self.margin) / self.cell_size))
                    if 0 <= x < self.board_size and 0 <= y < self.board_size:
                        if self.board.board[y, x] == 0:
                            return (y, x)
            pygame.time.wait(50)

    def show_winner(self, winner: int):
        font = pygame.font.SysFont(None, 48)
        if winner == 1:
            text = "Black wins"
        elif winner == -1:
            text = "White wins"
        else:
            text = "Draw"
        img = font.render(text, True, (255, 0, 0))
        rect = img.get_rect(center=(self.screen.get_width() // 2, self.margin // 2))
        self.screen.blit(img, rect)
        pygame.display.flip()

    def draw_board(self):
        self.screen.fill((205, 170, 125))
        start = self.margin
        end = self.margin + self.cell_size * (self.board_size - 1)
        for i in range(self.board_size):
            offset = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, (0, 0, 0), (start, offset), (end, offset), 1)
            pygame.draw.line(self.screen, (0, 0, 0), (offset, start), (offset, end), 1)
        for y in range(self.board_size):
            for x in range(self.board_size):
                p = self.board.board[y, x]
                if p != 0:
                    color = (0, 0, 0) if p == 1 else (255, 255, 255)
                    pos = (self.margin + x * self.cell_size, self.margin + y * self.cell_size)
                    pygame.draw.circle(self.screen, color, pos, self.cell_size // 2 - 2)
        if self.board.win_path:
            pts = [(
                self.margin + x * self.cell_size,
                self.margin + y * self.cell_size
            ) for y, x in self.board.win_path]
            pygame.draw.lines(self.screen, (255, 0, 0), False, pts, 3)
        pygame.display.flip()

    def play(self):
        current_player = 1
        while True:
            self.draw_board()
            if self.board.is_terminal():
                self.show_winner(self.board.winner())
                pygame.time.wait(2000)
                break

            if self.is_human_turn(current_player):
                res = self.get_human_move()
                if res == 'undo':
                    current_player = self.apply_undo()
                    continue
                move = res
            else:
                undone = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_u:
                        current_player = self.apply_undo()
                        undone = True
                        break
                if undone:
                    continue
                agent = self.agent_black if current_player == 1 else self.agent_white
                move = agent.select_move(self.board.copy(), current_player)
                pygame.time.wait(self.delay)

            self.board.step(move, current_player)
            current_player = -current_player


if __name__ == "__main__":
    # Example usage: human vs random agent
    model = PolicyValueNet()
    model.load_state_dict(torch.load("model_step160.pth"))
    modelAgent = MCTSAgent(model)
    game = PygameMatch(modelAgent, modelAgent)
    # game = PygameMatch(None, modelAgent)
    game.play()
