import chess
import chess.pgn
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import yaml

class ChessDataset(Dataset):
    def __init__(self, pgn_path, username):
        self.positions = []
        self.moves = []
        self.values = []
        
        try:
            with open(pgn_path, 'r') as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if not game:
                        break
                    
                    if game.headers["White"] == username or game.headers["Black"] == username:
                        board = game.board()
                        game_moves = list(game.mainline_moves())
                        result = game.headers.get("Result", "1/2-1/2")
                        
                        # Convert result to numeric value
                        if result == "1-0":
                            game_value = 1.0
                        elif result == "0-1":
                            game_value = -1.0
                        else:
                            game_value = 0.0
                        
                        for i, move in enumerate(game_moves):
                            if (board.turn == chess.WHITE and game.headers["White"] == username) or \
                               (board.turn == chess.BLACK and game.headers["Black"] == username):
                                
                                self.positions.append(self.board_to_tensor(board))
                                self.moves.append(move.uci())
                                
                                # Assign value based on game outcome and player color
                                player_value = game_value
                                if board.turn == chess.BLACK:
                                    player_value = -game_value
                                
                                self.values.append(player_value)
                            
                            board.push(move)
        except FileNotFoundError:
            print(f"Warning: PGN file {pgn_path} not found. Creating empty dataset.")
            
        print(f"Loaded {len(self.positions)} positions from dataset")

    def board_to_tensor(self, board):
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
        
        return tensor

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.positions[idx], dtype=torch.float32),
            self.moves[idx],
            torch.tensor(self.values[idx], dtype=torch.float32)
        )

class PolicyValueNetwork(nn.Module):
    def __init__(self, num_actions=4096, config=None):
        super().__init__()
        self.config = config or {}
        self.num_actions = num_actions
        
        # Shared convolutional backbone
        self.conv_layers = nn.Sequential(
            nn.Conv2d(12, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_actions)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.conv_layers(x)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value

class MoveEncoder:
    def __init__(self):
        self.move_to_index = {}
        self.index_to_move = {}
        self._build_move_mapping()
    
    def _build_move_mapping(self):
        move_list = []
        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq != to_sq:
                    move_list.append(chess.Move(from_sq, to_sq))
                    for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                        move_list.append(chess.Move(from_sq, to_sq, promotion=promotion))
        
        for i, move in enumerate(move_list):
            move_uci = move.uci()
            self.move_to_index[move_uci] = i
            self.index_to_move[i] = move_uci
    
    def encode_move(self, move_uci):
        return self.move_to_index.get(move_uci, 0)
    
    def decode_move(self, index):
        return self.index_to_move.get(index, "e2e4")
    
    def get_vocab_size(self):
        return len(self.move_to_index)

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default values.")
        return {}
