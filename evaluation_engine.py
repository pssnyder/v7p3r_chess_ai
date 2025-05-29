# evaluation_engine.py
import chess
import numpy as np

class EvaluationEngine:
    # Set depth for eval
    def __init__(self, board, depth=3):
        self.board = board
        self.depth = depth
        self.piece_values = {
            chess.KING: 10,
            chess.QUEEN: 9,
            chess.ROOK: 5,
            chess.BISHOP: 3,
            chess.KNIGHT: 3,
            chess.PAWN: 1
        }
    
    # Rules
        # _checkmate_threats()
        # _material_evaluation()
        # _positional_evaluation()
        # _king_safety()
        # _special_moves()
        # _check_threats()
        # _attack_evaluation()
        # _trapped_pieces()
        # _development()
        # _piece_value_modifiers()
        # _is_repeating_positions()
    
    def evaluate_position(self):
        # Calculate from both perspectives
        white_score = self._calculate_score(chess.WHITE)
        black_score = self._calculate_score(chess.BLACK)
        return white_score - black_score  # Net score favoring white
    
    def evaluate_position_with_lookahead(self):
        return self._minimax(self.depth, -float('inf'), float('inf'), True)

    def _minimax(self, depth, alpha, beta, maximizing_player):
        board = self.board.copy()  # Use copy instead of original
        if depth == 0 or self.board.is_game_over():
            return self.evaluate_position()
        
        if maximizing_player:
            max_eval = -float('inf')
            for move in self.board.legal_moves:
                self.board.push(move)
                eval = self._minimax(depth-1, alpha, beta, False)
                self.board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self._minimax(depth-1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def _calculate_score(self, color):
        score = 0
        score += self._material_score(color)
        score += self._piece_activity(color)
        score += self._king_safety(color)
        score += self._pawn_structure(color)
        score += self._center_control(color)
        return score
    
    # Rule Implementation
    # -------------------
    def _checkmate_threats(self):
        score = 0
        # Rule: Check for forced mate in 1
        for move in self.board.legal_moves:
            self.board.push(move)
            if self.board.is_checkmate():
                score += 1000 if self.board.turn else -1000
            self.board.pop()
        return score

    def _material_score(self, color):
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3.25,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King value handled in king safety
        }
        return sum(len(self.board.pieces(p, color)) * v for p, v in values.items())
    
    def _material_evaluation(self):
        """Calculate material advantage with phase-based piece values"""
        phase = self._game_phase()
        score = 0.0
        
        # Define piece values (opening, endgame)
        PIECE_VALUES = {
            chess.PAWN: (1.0, 1.2),
            chess.KNIGHT: (3.0, 3.2),
            chess.BISHOP: (3.2, 3.3),
            chess.ROOK: (5.0, 5.5),
            chess.QUEEN: (9.0, 9.5),
            chess.KING: (0, 4.0)  # Increased value in endgame
        }

        for piece_type in PIECE_VALUES:
            opening_val, endgame_val = PIECE_VALUES[piece_type]
            
            # Interpolate value based on game phase
            piece_value = opening_val * (1 - phase) + endgame_val * phase
            
            # Calculate material difference
            white_count = len(self.board.pieces(piece_type, chess.WHITE))
            black_count = len(self.board.pieces(piece_type, chess.BLACK))
            score += (white_count - black_count) * piece_value

        # Adjust score for current player's perspective
        return score if self.board.turn == chess.WHITE else -score
    
    def _game_phase(self):
        """Calculate game phase (0 = opening, 1 = endgame) based on remaining material"""
        # Count non-pawn material for both sides
        non_pawn_material = 0
        for piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            non_pawn_material += len(self.board.pieces(piece, chess.WHITE))
            non_pawn_material += len(self.board.pieces(piece, chess.BLACK))
        
        # Normalize to 0-1 range (assuming max 32 non-pawn pieces in initial position)
        phase = 1.0 - min(non_pawn_material / 32, 1.0)
        return phase

    def _positional_evaluation(self):
        score = 0
        # Rule: Center square control
        center = [chess.D4, chess.D5, chess.E4, chess.E5]
        for square in center:
            if self.board.piece_at(square) and self.board.piece_at(square).color == self.board.turn:
                score += 1
        return score

    def _king_safety(self):
        score = 0
        if self.board.is_check():
            score -= 10
            
        # Rule: Castling rights
        if self.board.has_castling_rights(self.board.turn):
            score += 3
            
        return score

    def _special_moves(self):
        score = 0
        # Rule: En passant
        if self.board.ep_square:
            score += 1
            
        # Rule: Promotion
        for move in self.board.legal_moves:
            if move.promotion:
                score += 5
        return score

    def _check_threats(self):
        score = 0
        # Current player's check threats
        for move in self.board.legal_moves:
            self.board.push(move)
            if self.board.is_check():
                score += 10
            self.board.pop()
            
        # Opponent's next-move check threats
        temp_board = self.board.copy()
        temp_board.turn = not self.board.turn
        if temp_board.is_check():
            score -= 5
        return score

    def _attack_evaluation(self):
        score = 0
        # Capturing higher-value pieces
        for move in self.board.legal_moves:
            if self.board.is_capture(move):
                attacker = self.board.piece_at(move.from_square)
                victim = self.board.piece_at(move.to_square)
                if victim and self.piece_values[victim.piece_type] > self.piece_values[attacker.piece_type]:
                    score += 3 + (self.piece_values[victim.piece_type] - self.piece_values[attacker.piece_type])

        # Hanging pieces and attacks
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                attackers = self.board.attackers(not self.board.turn, square)
                defenders = self.board.attackers(self.board.turn, square)
                
                if attackers:
                    max_attack = max(self.piece_values[self.board.piece_at(a).piece_type] for a in attackers)
                    if max_attack > self.piece_values[piece.piece_type] and not defenders:
                        score -= 3 + (max_attack - self.piece_values[piece.piece_type])
                        
                if not defenders and attackers:
                    score += 2
        return score

    def _trapped_pieces(self):
        score = 0
        legal_moves = list(self.board.legal_moves)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                moves = [m for m in legal_moves if m.from_square == square]
                if not moves or all(self.board.is_capture(m) for m in moves):
                    score -= 5
        return score

    def _development(self):
        bishops = knights = 0
        start_bishops = [chess.B1, chess.G1] if self.board.turn == chess.WHITE else [chess.B8, chess.G8]
        start_knights = [chess.B1, chess.G1] if self.board.turn == chess.WHITE else [chess.B8, chess.G8]
        
        for sq in start_bishops:
            if self.board.piece_at(sq) != chess.Piece(chess.BISHOP, self.board.turn):
                bishops += 1
        for sq in start_knights:
            if self.board.piece_at(sq) != chess.Piece(chess.KNIGHT, self.board.turn):
                knights += 1
                
        return 2 if bishops >= 2 and knights >= 2 else 0

    def _piece_value_modifiers(self):
        modifier = 0
        knights = []
        bishops = []
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                if piece.piece_type == chess.KNIGHT:
                    knights.append(square)
                elif piece.piece_type == chess.BISHOP:
                    if len(self.board.attacks(square)) > 3:
                        modifier += 1
                elif piece.piece_type == chess.PAWN and self._is_passed_pawn(square):
                    modifier += 1
                    
        if len(knights) >= 2:
            modifier += len(knights)
            
        return modifier

    def _is_passed_pawn(self, square):
        pawn = self.board.piece_at(square)
        if pawn.piece_type != chess.PAWN:
            return False
            
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        color = pawn.color
        start_rank = 1 if color == chess.WHITE else 6
        direction = 1 if color == chess.WHITE else -1
        
        # Check opposing pawns in front
        for f in [file-1, file, file+1]:
            if 0 <= f <= 7:
                for r in range(rank + direction, 7 if color == chess.WHITE else 0, direction):
                    target = chess.square(f, r)
                    opp_pawn = self.board.piece_at(target)
                    if opp_pawn and opp_pawn.piece_type == chess.PAWN and opp_pawn.color != color:
                        return False
        return True

    def _is_repeating_positions(self):
        score = 0
        
        # Penalize repeating positions
        if self.board.is_repetition(count=2):
            score -= 100  # Strong penalty to prevent draw by repetition
        
        return score
                