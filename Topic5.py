# -*- coding: utf-8 -*-
"""
Connect 4 using Reinforcement Learning
Modified from Tic Tac Toe RL structure
"""

import numpy as np

ROWS = 5
COLS = 5
WIN_LENGTH = 4


class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((ROWS, COLS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.playerSymbol = 1

    def getHash(self):
        return str(self.board.reshape(ROWS * COLS))

    def availablePositions(self):
        return [c for c in range(COLS) if self.board[0, c] == 0]

    def updateState(self, action):
        for r in reversed(range(ROWS)):
            if self.board[r, action] == 0:
                self.board[r, action] = self.playerSymbol
                break
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    def winner(self):
        # Horizontal, Vertical, Diagonal
        for r in range(ROWS):
            for c in range(COLS):
                if self.board[r, c] == 0:
                    continue
                if c + WIN_LENGTH <= COLS and abs(sum(self.board[r, c:c+WIN_LENGTH])) == WIN_LENGTH:
                    self.isEnd = True
                    return self.board[r, c]
                if r + WIN_LENGTH <= ROWS and abs(sum(self.board[r:r+WIN_LENGTH, c])) == WIN_LENGTH:
                    self.isEnd = True
                    return self.board[r, c]
                if r + WIN_LENGTH <= ROWS and c + WIN_LENGTH <= COLS:
                    diag1 = sum(self.board[r+i, c+i] for i in range(WIN_LENGTH))
                    diag2 = sum(self.board[r+WIN_LENGTH-1-i, c+i] for i in range(WIN_LENGTH))
                    if abs(diag1) == WIN_LENGTH or abs(diag2) == WIN_LENGTH:
                        self.isEnd = True
                        return self.board[r, c]

        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        return None

    def giveReward(self):
        result = self.winner()
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.3)
            self.p2.feedReward(0.3)

    def reset(self):
        self.board = np.zeros((ROWS, COLS))
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds=1500):
        for i in range(rounds):
            if i % 300 == 0:
                print(f"Training episode: {i}")
            while not self.isEnd:
                positions = self.availablePositions()
                action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(action)
                self.p1.addState(self.getHash())

                win = self.winner()
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                positions = self.availablePositions()
                action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(action)
                self.p2.addState(self.getHash())

                win = self.winner()
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

    def showBoard(self):
        print("\nBoard:")
        for r in range(ROWS):
            print("|", end="")
            for c in range(COLS):
                if self.board[r, c] == 1:
                    print(" X |", end="")
                elif self.board[r, c] == -1:
                    print(" O |", end="")
                else:
                    print("   |", end="")
            print()
        print("-" * (COLS * 4))

    def play_human(self):
        while not self.isEnd:
            positions = self.availablePositions()
            action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            self.updateState(action)
            self.showBoard()

            result = self.winner()
            if result is not None:
                print("Agent wins!" if result == 1 else "Draw!")
                self.reset()
                break

            positions = self.availablePositions()
            action = self.p2.chooseAction(positions)
            self.updateState(action)
            self.showBoard()

            result = self.winner()
            if result is not None:
                print("Human wins!" if result == -1 else "Draw!")
                self.reset()
                break


class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states = []
        self.states_value = {}

    def getHash(self, board):
        return str(board.reshape(ROWS * COLS))

    def chooseAction(self, positions, current_board, symbol):
        if np.random.rand() <= self.exp_rate:
            return np.random.choice(positions)

        value_max = -999
        for p in positions:
            next_board = current_board.copy()
            for r in reversed(range(ROWS)):
                if next_board[r, p] == 0:
                    next_board[r, p] = symbol
                    break
            value = self.states_value.get(self.getHash(next_board), 0)
            if value >= value_max:
                value_max = value
                action = p
        return action

    def addState(self, state):
        self.states.append(state)

    def feedReward(self, reward):
        for st in reversed(self.states):
            self.states_value[st] = self.states_value.get(st, 0)
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            col = int(input(f"Choose column {positions}: "))
            if col in positions:
                return col


if __name__ == "__main__":
    p1 = Player("Agent")
    p2 = Player("Opponent")

    game = State(p1, p2)
    print("Training Connect 4 agent...")
    game.play()
    print("Training complete.")

    human = HumanPlayer("Human")
    game = State(p1, human)

    print("\nPlay Connect 4 against the trained agent")
    game.showBoard()
    game.play_human()
