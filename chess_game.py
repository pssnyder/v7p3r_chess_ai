import sys
import os
import time
import chess
import chess.pgn
import torch
import numpy as np
import pygame
import random
from chess_core import PolicyValueNetwork, load_config, MoveEncoder
from evaluation_engine import EvaluationEngine

# Pygame constants
WIDTH, HEIGHT = 640, 640
DIMENSION = 8
SQ_SIZE = WIDTH // DIMENSION
MAX_FPS = 15
IMAGES = {}

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class ChessGame:
    def __init__(self, model_path="chess_rl_model.pth", config_path="config.yaml"):
        self.config = load_config(config_path)
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('v7p3r Chess RL AI')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.board = chess.Board()
        self.selected_square = None
        self.load_images()
        
        self.move_encoder = MoveEncoder()
        self.model = self._load_model(model_path)
        self.evaluator = EvaluationEngine(self.board, self.config)
        
        self.game = chess.pgn.Game()
        self.game_node = self.game
        self.ai_color = None
        self.human_color = None
        self.flip_board = False
        self.last_ai_move = None
        self.current_eval = 0.0
        self.thinking_time = 0.0

    def _load_model(self, model_path):
        model = PolicyValueNetwork(
            num_actions=self.move_encoder.get_vocab_size(),
            config=self.config
        ).to(self.device)
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}. Using random initialization.")
        else:
            print(f"Model file {model_path} not found. Using random initialization.")
        
        model.eval()
        return model

    def load_images(self):
        pieces = ['wp', 'wN', 'wb', 'wr', 'wq', 'wk',
                  'bp', 'bN', 'bb', 'br', 'bq', 'bk']
        for piece in pieces:
            try:
                IMAGES[piece] = pygame.transform.scale(
                    pygame.image.load(resource_path(f"images/{piece}.png")),
                    (SQ_SIZE, SQ_SIZE)
                )
            except pygame.error:
                print(f"Warning: Could not load image for {piece}")
                # Create a simple colored rectangle as fallback
                surface = pygame.Surface((SQ_SIZE, SQ_SIZE))
                color = (255, 255, 255) if piece[0] == 'w' else (0, 0, 0)
                surface.fill(color)
                IMAGES[piece] = surface

    def board_to_tensor(self, board):
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                row = 7 - (square // 8)
                col = square % 8
                tensor[channel][row][col] = 1
        return torch.tensor(tensor, device=self.device).unsqueeze(0)

    def ai_move(self):
        start_time = time.time()
        legal_moves = list(self.board.legal_moves)
        
        if not legal_moves:
            return None

        position_tensor = self.board_to_tensor(self.board)
        
        with torch.no_grad():
            policy_logits, value = self.model(position_tensor)
            legal_mask = torch.zeros(self.move_encoder.get_vocab_size(), device=self.device)
            
            for move in legal_moves:
                move_idx = self.move_encoder.encode_move(move.uci())
                legal_mask[move_idx] = 1.0
            
            policy_probs = torch.softmax(policy_logits, dim=1)
            masked_probs = policy_probs * legal_mask.unsqueeze(0)
            
            if masked_probs.sum() > 0:
                masked_probs /= masked_probs.sum()
            
            best_move_idx = torch.argmax(masked_probs).item()
            best_move_uci = self.move_encoder.decode_move(best_move_idx)
            
            try:
                move = chess.Move.from_uci(best_move_uci)
                if move in legal_moves:
                    self.current_eval = value.item()
                    self.thinking_time = time.time() - start_time
                    return best_move_uci
            except ValueError:
                pass
            
            # Fallback to random legal move
            fallback_move = random.choice(legal_moves)
            self.current_eval = value.item()
            self.thinking_time = time.time() - start_time
            return fallback_move.uci()

    def _setup_game(self):
        """Setup game parameters"""
        human_color_choice = self.config.get('game', {}).get('human_color', 'random')
        
        if human_color_choice == 'random':
            self.human_color = random.choice([chess.WHITE, chess.BLACK])
        elif human_color_choice == 'white':
            self.human_color = chess.WHITE
        elif human_color_choice == 'black':
            self.human_color = chess.BLACK
        else:
            self.human_color = chess.WHITE
        
        self.ai_color = not self.human_color
        self.flip_board = self.human_color == chess.BLACK
        
        # Set PGN headers
        self.game.headers["Event"] = "Human vs RL AI"
        self.game.headers["Site"] = "Local Computer"
        self.game.headers["Date"] = "???"
        self.game.headers["Round"] = "?"
        self.game.headers["White"] = "Human" if self.human_color == chess.WHITE else "v7p3r_RL_AI"
        self.game.headers["Black"] = "Human" if self.human_color == chess.BLACK else "v7p3r_RL_AI"
        self.game.headers["Result"] = "*"
        
        print(f"Game setup: Human plays {'White' if self.human_color == chess.WHITE else 'Black'}")

    def _handle_mouse_click(self, pos):
        """Handle mouse clicks for piece selection and movement"""
        if self.board.turn != self.human_color or self.board.is_game_over():
            return
        
        col = pos[0] // SQ_SIZE
        row = pos[1] // SQ_SIZE
        
        if self.flip_board:
            col = 7 - col
            row = 7 - row
        
        square = chess.square(col, 7 - row)
        
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
        else:
            move = chess.Move(self.selected_square, square)
            
            # Check for pawn promotion
            piece = self.board.piece_at(self.selected_square)
            if (piece and piece.piece_type == chess.PAWN and 
                ((chess.square_rank(square) == 7 and self.board.turn == chess.WHITE) or
                 (chess.square_rank(square) == 0 and self.board.turn == chess.BLACK))):
                move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
            
            if move in self.board.legal_moves:
                self.board.push(move)
                self.game_node = self.game_node.add_variation(move)
                print(f"Human plays: {move.uci()}")
            
            self.selected_square = None

    def _process_ai_move(self):
        """Process AI move"""
        try:
            ai_move_uci = self.ai_move()
            if ai_move_uci:
                move = chess.Move.from_uci(ai_move_uci)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.game_node = self.game_node.add_variation(move)
                    self.last_ai_move = move.to_square
                    
                    print(f"AI plays: {ai_move_uci} (Eval: {self.current_eval:.2f}, "
                          f"Time: {self.thinking_time:.2f}s)")
                else:
                    print(f"AI suggested illegal move: {ai_move_uci}")
        except Exception as e:
            print(f"AI move error: {e}")

    def _update_display(self):
        """Update the game display"""
        self.draw_board()
        self.draw_pieces()
        self.draw_highlights()
        self.draw_legal_moves()
        self.draw_info()
        
        pygame.display.flip()

    def draw_board(self):
        """Draw the chess board"""
        colors = [pygame.Color("#f0d9b5"), pygame.Color("#b58863")]
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                draw_r = 7 - r if self.flip_board else r
                draw_c = 7 - c if self.flip_board else c
                color = colors[(draw_r + draw_c) % 2]
                pygame.draw.rect(self.screen, color,
                               pygame.Rect(draw_c*SQ_SIZE, draw_r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def draw_pieces(self):
        """Draw the chess pieces"""
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                board_r = 7 - r if self.flip_board else r
                board_c = 7 - c if self.flip_board else c
                square = chess.square(board_c, 7 - board_r)
                piece = self.board.piece_at(square)
                
                if piece:
                    color = 'w' if piece.color == chess.WHITE else 'b'
                    piece_type = piece.symbol().upper()
                    img_key = f"{color}{piece_type.lower()}" if piece_type != 'N' else f"{color}N"
                    
                    if img_key in IMAGES:
                        self.screen.blit(IMAGES[img_key],
                                       pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def draw_highlights(self):
        """Draw square highlights"""
        # Highlight selected square
        if self.selected_square is not None:
            file = chess.square_file(self.selected_square)
            rank = chess.square_rank(self.selected_square)
            
            if self.flip_board:
                file = 7 - file
                rank = 7 - rank
            
            s = pygame.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(150)
            s.fill(pygame.Color('blue'))
            self.screen.blit(s, (file*SQ_SIZE, (7-rank)*SQ_SIZE))
        
        # Highlight AI's last move
        if self.last_ai_move is not None:
            file = chess.square_file(self.last_ai_move)
            rank = chess.square_rank(self.last_ai_move)
            
            if self.flip_board:
                file = 7 - file
                rank = 7 - rank
            
            s = pygame.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)
            s.fill(pygame.Color('yellow'))
            self.screen.blit(s, (file*SQ_SIZE, (7-rank)*SQ_SIZE))

    def draw_legal_moves(self):
        """Draw legal move indicators"""
        if self.selected_square is not None:
            for move in self.board.legal_moves:
                if move.from_square == self.selected_square:
                    file = chess.square_file(move.to_square)
                    rank = chess.square_rank(move.to_square)
                    
                    if self.flip_board:
                        file = 7 - file
                        rank = 7 - rank
                    
                    center_x = file * SQ_SIZE + SQ_SIZE // 2
                    center_y = (7 - rank) * SQ_SIZE + SQ_SIZE // 2
                    
                    pygame.draw.circle(self.screen, pygame.Color('green'),
                                     (center_x, center_y), 8)

    def draw_info(self):
        """Draw game information"""
        # Draw evaluation
        eval_text = f"Eval: {self.current_eval:+.2f}"
        color = (255, 255, 255)
        if self.current_eval > 0.5:
            color = (0, 255, 0)
        elif self.current_eval < -0.5:
            color = (255, 0, 0)
        
        text = self.font.render(eval_text, True, color)
        self.screen.blit(text, (10, 10))
        
        # Draw whose turn it is
        turn_text = f"{'White' if self.board.turn == chess.WHITE else 'Black'} to move"
        text = self.font.render(turn_text, True, (255, 255, 255))
        self.screen.blit(text, (10, 40))
        
        # Draw thinking time
        if self.thinking_time > 0:
            time_text = f"Think time: {self.thinking_time:.2f}s"
            text = self.font.render(time_text, True, (255, 255, 255))
            self.screen.blit(text, (10, 70))

    def _handle_game_end(self):
        """Handle game end conditions"""
        if self.board.is_game_over():
            result = self.board.result()
            reason = "Checkmate"
            
            if self.board.is_stalemate():
                reason = "Stalemate"
            elif self.board.is_insufficient_material():
                reason = "Insufficient material"
            elif self.board.is_seventyfive_moves():
                reason = "75-move rule"
            elif self.board.is_fivefold_repetition():
                reason = "Fivefold repetition"
            
            print(f"\nGame over: {result} ({reason})")
            self.game.headers["Result"] = result
            
            # Save PGN
            self.save_pgn()
            
            # Wait for user input before closing
            print("Press any key to close...")
            pygame.event.clear()
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                        waiting = False
            
            return True
        return False

    def save_pgn(self, filename="rl_ai_game.pgn"):
        """Save the game as PGN"""
        try:
            with open(filename, "w") as f:
                exporter = chess.pgn.FileExporter(f)
                self.game.accept(exporter)
            print(f"Game saved to {filename}")
        except Exception as e:
            print(f"Error saving PGN: {e}")

    def run(self):
        """Main game loop"""
        self._setup_game()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_click(pygame.mouse.get_pos())
            
            # Handle AI move
            if self.board.turn == self.ai_color and not self.board.is_game_over():
                self._process_ai_move()
            
            # Update display
            self._update_display()
            
            # Check for game end
            if self._handle_game_end():
                running = False
            
            self.clock.tick(MAX_FPS)
        
        pygame.quit()

if __name__ == "__main__":
    game = ChessGame()
    game.run()