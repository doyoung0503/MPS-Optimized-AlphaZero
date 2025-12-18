"""
Simplified Chess Engine for AlphaZero Training
8x8 board, standard chess rules
"""
import numpy as np
from copy import deepcopy

# Piece encoding: 1=P, 2=N, 3=B, 4=R, 5=Q, 6=K (positive=White, negative=Black)

class Chess:
    def __init__(self):
        self.board = self._initial_board()
        self.turn = 1  # 1=White, -1=Black
        self.fullmoves = 0
        
    def _initial_board(self):
        """Standard chess starting position"""
        board = np.zeros((8, 8), dtype=np.int8)
        # Black pieces (negative)
        board[0] = [-4, -2, -3, -5, -6, -3, -2, -4]
        board[1] = [-1, -1, -1, -1, -1, -1, -1, -1]
        # White pieces (positive)
        board[6] = [1, 1, 1, 1, 1, 1, 1, 1]
        board[7] = [4, 2, 3, 5, 6, 3, 2, 4]
        return board
    
    def get_legal_moves(self):
        """Get all legal moves as (from_sq, to_sq) tuples"""
        moves = []
        for r in range(8):
            for c in range(8):
                sq = r * 8 + c
                piece = self.board[r, c]
                if piece * self.turn > 0:  # Our piece
                    moves.extend(self._get_piece_moves(r, c, piece))
        return moves
    
    def _get_piece_moves(self, r, c, piece):
        """Get moves for a specific piece"""
        moves = []
        p_type = abs(piece)
        
        if p_type == 1:  # Pawn
            direction = -1 if piece > 0 else 1
            # Forward
            nr = r + direction
            if 0 <= nr < 8 and self.board[nr, c] == 0:
                moves.append((r*8+c, nr*8+c))
                # Double push from start
                if (piece > 0 and r == 6) or (piece < 0 and r == 1):
                    nr2 = r + 2*direction
                    if self.board[nr2, c] == 0:
                        moves.append((r*8+c, nr2*8+c))
            # Captures
            for dc in [-1, 1]:
                nc = c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    target = self.board[nr, nc]
                    if target * piece < 0:  # Enemy piece
                        moves.append((r*8+c, nr*8+nc))
        
        elif p_type == 2:  # Knight
            for dr, dc in [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    target = self.board[nr, nc]
                    if target * piece <= 0:  # Empty or enemy
                        moves.append((r*8+c, nr*8+nc))
        
        elif p_type in [3, 4, 5]:  # Bishop, Rook, Queen
            directions = []
            if p_type in [3, 5]:  # Bishop or Queen
                directions += [(1,1),(1,-1),(-1,1),(-1,-1)]
            if p_type in [4, 5]:  # Rook or Queen
                directions += [(1,0),(-1,0),(0,1),(0,-1)]
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                while 0 <= nr < 8 and 0 <= nc < 8:
                    target = self.board[nr, nc]
                    if target == 0:
                        moves.append((r*8+c, nr*8+nc))
                    elif target * piece < 0:  # Enemy
                        moves.append((r*8+c, nr*8+nc))
                        break
                    else:  # Our piece
                        break
                    nr += dr
                    nc += dc
        
        elif p_type == 6:  # King
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        target = self.board[nr, nc]
                        if target * piece <= 0:
                            moves.append((r*8+c, nr*8+nc))
        
        return moves
    
    def push(self, move):
        """Make a move: (from_sq, to_sq)"""
        from_sq, to_sq = move
        fr, fc = from_sq // 8, from_sq % 8
        tr, tc = to_sq // 8, to_sq % 8
        
        piece = self.board[fr, fc]
        self.board[tr, tc] = piece
        self.board[fr, fc] = 0
        
        # Pawn promotion (simplified: always promote to Queen)
        if abs(piece) == 1 and (tr == 0 or tr == 7):
            self.board[tr, tc] = 5 if piece > 0 else -5
        
        self.turn *= -1
        self.fullmoves += 1
    
    def is_game_over(self):
        """Check if game is over (simplified: no legal moves or 200 moves)"""
        if self.fullmoves >= 200:
            return True
        return len(self.get_legal_moves()) == 0
    
    def get_winner(self):
        """Get winner: 1=White, -1=Black, 0=Draw"""
        if self.fullmoves >= 200:
            return 0
        
        # Check if current player has King
        has_king = False
        for r in range(8):
            for c in range(8):
                if self.board[r, c] * self.turn == 6:
                    has_king = True
                    break
            if has_king:
                break
        
        if not has_king:
            return -self.turn  # Other player won
        
        if len(self.get_legal_moves()) == 0:
            return -self.turn  # Checkmate/Stalemate -> opponent wins
        
        return 2  # Game not over
    
    def get_canonical(self):
        """Get canonical form (always from White's perspective)"""
        if self.turn == 1:
            return self
        else:
            # Flip board for Black
            return self.mirror()
    
    def mirror(self):
        """Mirror the board (flip vertically and negate pieces)"""
        mirrored = Chess()
        mirrored.board = -np.flipud(self.board)
        mirrored.turn = -self.turn
        mirrored.fullmoves = self.fullmoves
        return mirrored
    
    def encode(self):
        """Encode board state as 14x8x8 tensor (7 piece types x 2 players)"""
        state = np.zeros((14, 8, 8), dtype=np.float32)
        for r in range(8):
            for c in range(8):
                piece = self.board[r, c]
                if piece != 0:
                    p_type = abs(piece) - 1  # 0-5 for P,N,B,R,Q,K
                    player_offset = 0 if piece > 0 else 7
                    state[player_offset + p_type, r, c] = 1.0
        return state
    
    def copy(self):
        """Deep copy of game state"""
        new_game = Chess()
        new_game.board = self.board.copy()
        new_game.turn = self.turn
        new_game.fullmoves = self.fullmoves
        return new_game
