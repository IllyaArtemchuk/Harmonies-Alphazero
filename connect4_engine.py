import numpy as np
import copy
from typing import List, Tuple, Optional, Union

class Connect4GameState:
    def __init__(self, initial_state=None):
        if initial_state:
            self.__dict__.update(initial_state)
        else:
            self.board = np.zeros((6, 7), dtype=np.int8)
            self.current_player = 0
            self.game_over = False
            self.winner = None
            self.move_history = []
            self.id = None
    
    def __hash__(self):
        return hash((self.board.tobytes(), self.current_player))
    
    def clone(self):
        new_state = Connect4GameState()
        new_state.board = self.board.copy()
        new_state.current_player = self.current_player
        new_state.game_over = self.game_over
        new_state.winner = self.winner
        new_state.move_history = self.move_history.copy()
        new_state.id = self.id
        return new_state
    
    def get_current_player(self) -> int:
        return self.current_player
    
    def get_legal_moves(self) -> List[int]:
        if self.game_over:
            return []
        
        legal_moves = []
        for col in range(7):
            if self.board[0, col] == 0:
                legal_moves.append(col)
        return legal_moves
    
    def apply_move(self, move: int) -> 'Connect4GameState':
        if move not in self.get_legal_moves():
            raise ValueError(f"Invalid move: {move}")
        
        new_state = self.clone()
        
        for row in range(5, -1, -1):
            if new_state.board[row, move] == 0:
                new_state.board[row, move] = new_state.current_player + 1
                new_state.move_history.append((new_state.current_player, move))
                break
        
        new_state._check_win_condition()
        
        if not new_state.game_over:
            new_state.current_player = 1 - new_state.current_player
        
        return new_state
    
    def _check_win_condition(self):
        for row in range(6):
            for col in range(7):
                if self.board[row, col] != 0:
                    if self._check_direction(row, col, 0, 1) or \
                       self._check_direction(row, col, 1, 0) or \
                       self._check_direction(row, col, 1, 1) or \
                       self._check_direction(row, col, 1, -1):
                        self.game_over = True
                        self.winner = self.board[row, col] - 1
                        return
        
        if np.all(self.board != 0):
            self.game_over = True
            self.winner = None
    
    def _check_direction(self, row: int, col: int, delta_row: int, delta_col: int) -> bool:
        player = self.board[row, col]
        count = 1
        
        for i in range(1, 4):
            r, c = row + i * delta_row, col + i * delta_col
            if 0 <= r < 6 and 0 <= c < 7 and self.board[r, c] == player:
                count += 1
            else:
                break
        
        for i in range(1, 4):
            r, c = row - i * delta_row, col - i * delta_col
            if 0 <= r < 6 and 0 <= c < 7 and self.board[r, c] == player:
                count += 1
            else:
                break
        
        return count >= 4
    
    def is_game_over(self) -> bool:
        return self.game_over
    
    def get_game_outcome(self) -> int:
        if not self.game_over:
            return 0
        if self.winner is None:
            return 0
        return 1 if self.winner == 0 else -1
    
    def get_canonical_tuple(self) -> tuple:
        return (
            tuple(self.board.flatten()),
            self.current_player,
            self.game_over,
            self.winner,
            tuple(self.move_history)
        )
    
    def render(self, logger=None):
        symbols = [' ', 'X', 'O']
        lines = []
        lines.append("  0 1 2 3 4 5 6")
        lines.append(" +" + "-" * 13 + "+")
        
        for row in range(6):
            line = f"{row}|"
            for col in range(7):
                line += symbols[self.board[row, col]] + "|"
            lines.append(line)
        
        lines.append(" +" + "-" * 13 + "+")
        lines.append(f"Current player: {self.current_player} ({symbols[self.current_player + 1]})")
        
        if self.game_over:
            if self.winner is None:
                lines.append("Game Over: Draw!")
            else:
                lines.append(f"Game Over: Player {self.winner} ({symbols[self.winner + 1]}) wins!")
        
        output = "\n".join(lines)
        
        if logger:
            logger.info(output)
        else:
            print(output)
    
    def __str__(self):
        lines = []
        self.render()
        return ""