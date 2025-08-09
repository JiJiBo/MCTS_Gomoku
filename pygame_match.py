import sys
import time
import random
import numpy as np
import pygame

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


class PygameMatch:
    """Display a real-time match between two agents using pygame."""
    def __init__(self, agent_black: Agent, agent_white: Agent, board_size: int = 15,
                 cell_size: int = 40, margin: int = 20, delay: int = 500):
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
        pygame.display.flip()

    def play(self):
        current_player = 1
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.draw_board()
            if self.board.is_terminal():
                time.sleep(2)
                break
            agent = self.agent_black if current_player == 1 else self.agent_white
            move = agent.select_move(self.board.copy(), current_player)
            self.board.step(move, current_player)
            current_player = -current_player
            self.draw_board()
            pygame.time.wait(self.delay)


if __name__ == "__main__":
    # Example usage with two random agents
    game = PygameMatch(RandomAgent(), RandomAgent())
    game.play()
