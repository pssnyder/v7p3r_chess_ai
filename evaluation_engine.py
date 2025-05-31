import chess
import numpy as np

class EvaluationEngine:
    def __init__(self, board, config=None):
        self.board = board.copy() if hasattr(board, 'copy') else board
        self.config = config.get('evaluation', {}) if config else {}
        self.piece_values = {
            chess.KING: 10,
            chess.QUEEN: 9,
            chess.ROOK: 5,
            chess.BISHOP: 3,
            chess.KNIGHT: 3,
            chess.PAWN: 1
        }

    def evaluate_position(self):
        """Main evaluation function"""
        score = 0
        
        # Check for immediate wins
        if self.board.is_checkmate():
            return 1000 if self.board.turn else -1000
        
        # Check for forced mates
        score += self._checkmate_threats()
        
        # Material and positional evaluation
        score += self._material_evaluation()
        score += self._positional_evaluation()
        score += self._king_safety()
        score += self._special_moves()
        score += self._tactical_evaluation()
        
        return score

    def calculate_shaped_reward(self):
        """Calculate shaped reward for reinforcement learning"""
        reward = 0.0
        
        # Material evaluation with config weights
        reward += self._material_evaluation() * self.config.get('material_weight', 0.8)
        
        # Positional bonuses
        reward += self._center_control() * self.config.get('center_control_bonus', 0.5)
        reward += self._piece_activity() * self.config.get('piece_activity_bonus', 0.1)
        reward += self._king_safety_shaped() * self.config.get('king_safety_bonus', 0.3)
        
        # Tactical bonuses
        if self.board.is_check():
            reward += self.config.get('check_bonus', 10)
        
        # Development bonuses
        reward += self._piece_development() * self.config.get('piece_development_bonus', 2)
        
        # Special move bonuses
        reward += self._castling_evaluation() * self.config.get('castling_bonus', 5)
        reward += self._pawn_structure() * self.config.get('passed_pawn_bonus', 0.25)
        
        return reward

    def _checkmate_threats(self):
        """Check for mate in 1 threats"""
        score = 0
        checkmate_bonus = self.config.get('checkmate_bonus', 1000.0)
        
        # Check for mate in 1
        for move in self.board.legal_moves:
            self.board.push(move)
            if self.board.is_checkmate():
                score += checkmate_bonus if self.board.turn else -checkmate_bonus
            self.board.pop()
        
        return score

    def _material_evaluation(self):
        """Calculate material balance"""
        score = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        return score

    def _positional_evaluation(self):
        """Basic positional evaluation"""
        score = 0
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_score = 0
                
                # Center control
                if square in center_squares:
                    piece_score += self.config.get('center_control_bonus', 0.5)
                
                # Development bonus for minor pieces
                if piece.piece_type in [chess.BISHOP, chess.KNIGHT]:
                    starting_squares = [chess.B1, chess.G1, chess.B8, chess.G8] if piece.piece_type == chess.KNIGHT else [chess.C1, chess.F1, chess.C8, chess.F8]
                    if square not in starting_squares:
                        piece_score += self.config.get('piece_development_bonus', 2)
                
                # Apply score based on piece color
                if piece.color == chess.WHITE:
                    score += piece_score
                else:
                    score -= piece_score
        
        return score

    def _center_control(self):
        """Evaluate center square control"""
        score = 0
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        
        for square in center_squares:
            piece = self.board.piece_at(square)
            if piece:
                if piece.color == chess.WHITE:
                    score += 1
                else:
                    score -= 1
        
        return score

    def _piece_activity(self):
        """Evaluate piece mobility and activity"""
        score = 0
        
        # Knight activity
        for square in self.board.pieces(chess.KNIGHT, chess.WHITE):
            attacks = len(self.board.attacks(square))
            score += attacks * self.config.get('knight_activity_bonus', 0.1)
        
        for square in self.board.pieces(chess.KNIGHT, chess.BLACK):
            attacks = len(self.board.attacks(square))
            score -= attacks * self.config.get('knight_activity_bonus', 0.1)
        
        # Bishop activity
        for square in self.board.pieces(chess.BISHOP, chess.WHITE):
            attacks = len(self.board.attacks(square))
            score += attacks * self.config.get('bishop_activity_bonus', 0.15)
        
        for square in self.board.pieces(chess.BISHOP, chess.BLACK):
            attacks = len(self.board.attacks(square))
            score -= attacks * self.config.get('bishop_activity_bonus', 0.15)
        
        return score

    def _king_safety(self):
        """Basic king safety evaluation"""
        score = 0
        
        # Check threats
        if self.board.is_check():
            score += self.config.get('in_check_penalty', -10)
        
        # Castling rights
        if self.board.has_castling_rights(chess.WHITE):
            score += self.config.get('castling_bonus', 5)
        if self.board.has_castling_rights(chess.BLACK):
            score -= self.config.get('castling_bonus', 5)
        
        return score

    def _king_safety_shaped(self):
        """King safety evaluation for reward shaping"""
        score = 0
        
        # Pawn shield evaluation
        for color in [chess.WHITE, chess.BLACK]:
            king_square = self.board.king(color)
            if king_square is not None:
                pawn_shield = 0
                king_file = chess.square_file(king_square)
                king_rank = chess.square_rank(king_square)
                
                # Check for pawn shield
                shield_squares = []
                if color == chess.WHITE and king_rank < 7:
                    shield_squares = [king_square + 8 + i for i in [-1, 0, 1] if 0 <= king_file + i < 8]
                elif color == chess.BLACK and king_rank > 0:
                    shield_squares = [king_square - 8 + i for i in [-1, 0, 1] if 0 <= king_file + i < 8]
                
                for sq in shield_squares:
                    if 0 <= sq < 64:
                        piece = self.board.piece_at(sq)
                        if piece and piece.piece_type == chess.PAWN and piece.color == color:
                            pawn_shield += 1
                
                shield_bonus = pawn_shield * self.config.get('king_safety_bonus', 0.3)
                if color == chess.WHITE:
                    score += shield_bonus
                else:
                    score -= shield_bonus
        
        return score

    def _piece_development(self):
        """Evaluate piece development"""
        score = 0
        
        # Check if minor pieces are developed
        starting_squares = {
            chess.WHITE: [chess.B1, chess.G1, chess.C1, chess.F1],
            chess.BLACK: [chess.B8, chess.G8, chess.C8, chess.F8]
        }
        
        for color in [chess.WHITE, chess.BLACK]:
            developed = 0
            total_pieces = 0
            
            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                for square in self.board.pieces(piece_type, color):
                    total_pieces += 1
                    if square not in starting_squares[color]:
                        developed += 1
            
            if total_pieces > 0:
                development_ratio = developed / total_pieces
                development_score = development_ratio * self.config.get('piece_development_bonus', 2)
                
                if color == chess.WHITE:
                    score += development_score
                else:
                    score -= development_score
        
        return score

    def _special_moves(self):
        """Evaluate special moves and opportunities"""
        score = 0
        
        # En passant
        if self.board.ep_square:
            score += self.config.get('en_passant_bonus', 1)
        
        # Promotion opportunities
        for move in self.board.legal_moves:
            if move.promotion:
                score += self.config.get('pawn_promotion_bonus', 5)
        
        return score

    def _tactical_evaluation(self):
        """Evaluate tactical elements"""
        score = 0
        
        # Captures
        for move in self.board.legal_moves:
            if self.board.is_capture(move):
                score += self.config.get('capture_bonus', 3)
        
        # Checks
        for move in self.board.legal_moves:
            self.board.push(move)
            if self.board.is_check():
                score += self.config.get('check_bonus', 10)
            self.board.pop()
        
        return score

    def _castling_evaluation(self):
        """Evaluate castling rights and opportunities"""
        score = 0
        
        if self.board.has_castling_rights(chess.WHITE):
            score += 1
        if self.board.has_castling_rights(chess.BLACK):
            score -= 1
        
        return score

    def _pawn_structure(self):
        """Basic pawn structure evaluation"""
        score = 0
        
        # Check for passed pawns
        for color in [chess.WHITE, chess.BLACK]:
            for square in self.board.pieces(chess.PAWN, color):
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                
                # Simple passed pawn detection
                is_passed = True
                direction = 1 if color == chess.WHITE else -1
                
                # Check if there are opponent pawns blocking or attacking
                for check_rank in range(rank + direction, 8 if color == chess.WHITE else -1, direction):
                    if 0 <= check_rank < 8:
                        for check_file in [file - 1, file, file + 1]:
                            if 0 <= check_file < 8:
                                check_square = chess.square(check_file, check_rank)
                                piece = self.board.piece_at(check_square)
                                if piece and piece.piece_type == chess.PAWN and piece.color != color:
                                    is_passed = False
                                    break
                    if not is_passed:
                        break
                
                if is_passed:
                    passed_bonus = self.config.get('passed_pawn_bonus', 0.25)
                    if color == chess.WHITE:
                        score += passed_bonus
                    else:
                        score -= passed_bonus
        
        return score