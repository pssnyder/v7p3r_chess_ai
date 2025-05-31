import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
import random
from chess_core import PolicyValueNetwork, MoveEncoder, load_config
from evaluation_engine import EvaluationEngine

class ChessRLTrainer:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.move_encoder = MoveEncoder()
        
        self.policy_net = PolicyValueNetwork(
            num_actions=self.move_encoder.get_vocab_size(),
            config=self.config
        ).to(self.device)
        
        self.target_net = PolicyValueNetwork(
            num_actions=self.move_encoder.get_vocab_size(),
            config=self.config
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.get('training', {}).get('learning_rate', 0.001),
            weight_decay=self.config.get('training', {}).get('weight_decay', 1e-4)
        )
        
        self.criterion_policy = nn.CrossEntropyLoss()
        self.criterion_value = nn.MSELoss()
        
        print(f"Initialized trainer with {self.move_encoder.get_vocab_size()} possible moves")

    def _board_to_tensor(self, board):
        """Convert board to 12-channel tensor representation"""
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Channels 0-5: White pieces (P,N,B,R,Q,K)
                # Channels 6-11: Black pieces (P,N,B,R,Q,K)
                channel = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                row = 7 - (square // 8)  # Flip for proper orientation
                col = square % 8
                tensor[channel][row][col] = 1
        
        return torch.tensor(tensor, device=self.device).unsqueeze(0)

    def _select_move(self, policy_logits, board):
        """Select move using policy network with exploration"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Create legal move mask
        legal_mask = torch.zeros(self.move_encoder.get_vocab_size(), device=self.device)
        legal_indices = []
        
        for move in legal_moves:
            move_idx = self.move_encoder.encode_move(move.uci())
            legal_mask[move_idx] = 1.0
            legal_indices.append((move, move_idx))
        
        # Apply softmax and mask
        policy_probs = torch.softmax(policy_logits, dim=1)
        masked_probs = policy_probs * legal_mask.unsqueeze(0)
        
        if masked_probs.sum() > 0:
            masked_probs /= masked_probs.sum()
        
        # Add exploration noise
        exploration_rate = self.config.get('training', {}).get('exploration_rate', 0.1)
        if random.random() < exploration_rate:
            return random.choice(legal_moves)
        
        # Select best legal move
        best_move_idx = torch.argmax(masked_probs).item()
        best_move_uci = self.move_encoder.decode_move(best_move_idx)
        
        try:
            move = chess.Move.from_uci(best_move_uci)
            if move in legal_moves:
                return move
        except ValueError:
            pass
        
        # Fallback to random legal move
        return random.choice(legal_moves)

    def _calculate_game_outcome(self, board):
        """Calculate game outcome value"""
        if board.is_checkmate():
            return 1.0 if board.turn == chess.BLACK else -1.0
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        return None  # Game not finished

    def _get_reward_shaping(self, board):
        """Calculate shaped reward using evaluation engine"""
        try:
            evaluator = EvaluationEngine(board)
            eval_config = self.config.get('evaluation', {})
            
            reward = 0.0
            
            # Check for checkmate threats
            if board.is_check():
                reward += eval_config.get('check_bonus', 10)
            
            # Material evaluation
            material_score = 0
            piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                          chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
            
            for piece_type, value in piece_values.items():
                white_count = len(board.pieces(piece_type, chess.WHITE))
                black_count = len(board.pieces(piece_type, chess.BLACK))
                material_score += (white_count - black_count) * value
            
            reward += material_score * eval_config.get('material_weight', 0.8)
            
            # Center control
            center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
            for square in center_squares:
                if board.piece_at(square):
                    reward += eval_config.get('center_control_bonus', 0.5)
            
            return reward / 100.0  # Scale reward
            
        except Exception as e:
            print(f"Warning: Error in reward shaping: {e}")
            return 0.0

    def self_play(self, num_games=100):
        """Generate self-play games for training"""
        print(f"Starting self-play training with {num_games} games...")
        
        for game_idx in range(num_games):
            board = chess.Board()
            game_history = []
            move_count = 0
            max_moves = 200  # Prevent infinite games
            
            while not board.is_game_over() and move_count < max_moves:
                # Get policy and value from network
                position_tensor = self._board_to_tensor(board)
                
                with torch.no_grad():
                    policy_logits, value = self.policy_net(position_tensor)
                
                # Select move
                move = self._select_move(policy_logits, board)
                if move is None:
                    break
                
                # Store position, policy target, and initial value
                move_idx = self.move_encoder.encode_move(move.uci())
                game_history.append({
                    'position': position_tensor.clone(),
                    'move_idx': move_idx,
                    'value_pred': value.item(),
                    'turn': board.turn
                })
                
                # Make move
                board.push(move)
                move_count += 1
            
            # Calculate final game outcome
            game_outcome = self._calculate_game_outcome(board)
            if game_outcome is None:
                game_outcome = 0.0  # Draw for incomplete games
            
            # Update game history with final outcomes
            for i, entry in enumerate(game_history):
                # Assign outcome based on player perspective
                if entry['turn'] == chess.WHITE:
                    entry['target_value'] = game_outcome
                else:
                    entry['target_value'] = -game_outcome
                
                # Add reward shaping
                shaped_reward = self._get_reward_shaping(board)
                entry['target_value'] += shaped_reward
            
            # Train on this game
            if len(game_history) > 10:  # Only train on reasonable length games
                self.train_step(game_history)
            
            if (game_idx + 1) % 10 == 0:
                print(f"Completed game {game_idx + 1}/{num_games} "
                      f"(moves: {len(game_history)}, outcome: {game_outcome:.2f})")

    def train_step(self, game_history):
        """Train the network on a single game"""
        self.policy_net.train()
        
        batch_losses = []
        
        for entry in game_history:
            self.optimizer.zero_grad()
            
            # Forward pass
            policy_logits, value_pred = self.policy_net(entry['position'])
            
            # Create target for policy (one-hot encoded move)
            policy_target = torch.zeros(self.move_encoder.get_vocab_size(), device=self.device)
            policy_target[entry['move_idx']] = 1.0
            
            # Calculate losses
            policy_loss = self.criterion_policy(policy_logits, policy_target.unsqueeze(0))
            value_loss = self.criterion_value(value_pred, 
                                            torch.tensor([[entry['target_value']]], 
                                                       device=self.device, dtype=torch.float32))
            
            total_loss = policy_loss + value_loss
            batch_losses.append(total_loss.item())
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
        
        # Update target network periodically
        if len(batch_losses) > 0 and random.random() < 0.1:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filename="chess_rl_model.pth"):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename="chess_rl_model.pth"):
        """Load a trained model"""
        if torch.cuda.is_available():
            checkpoint = torch.load(filename)
        else:
            checkpoint = torch.load(filename, map_location='cpu')
        
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filename}")

if __name__ == "__main__":
    trainer = ChessRLTrainer()
    trainer.self_play(num_games=100)  # Start with smaller number for testing
    trainer.save_model("chess_rl_model.pth")
    print("Training completed!")