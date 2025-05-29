# chess_core.py
from evaluation_engine import EvaluationEngine
import chess
import chess.pgn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ChessDataset(Dataset):
    def __init__(self, pgn_path, username):
        self.positions = []
        self.moves = []
        
        pgn = open(pgn_path)
        while True:
            game = chess.pgn.read_game(pgn)
            if not game:
                break
            
            if game.headers["White"] == username or game.headers["Black"] == username:
                board = game.board()
                for move in game.mainline_moves():
                    if (board.turn == chess.WHITE and game.headers["White"] == username) or \
                       (board.turn == chess.BLACK and game.headers["Black"] == username):
                        self.positions.append(self.board_to_tensor(board))
                        self.moves.append(move.uci())
                    board.push(move)

    def board_to_tensor(self, board):
        tensor = np.zeros((13, 8, 8), dtype=np.float32)  # Changed from 12 to 13
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                tensor[channel][7 - square//8][square%8] = 1
        
        # Add promotion flag to pawns on 7th/2nd ranks
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                rank = chess.square_rank(square)
                if rank in [1, 6]:  # 2nd/7th ranks
                    tensor[12][7 - square//8][square%8] = 1
        return tensor

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.moves[idx]

class ChessAI(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv2d(13, 256, kernel_size=3, padding=1)  # Changed from 12 to 13
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256*8*8, num_classes)
        
    def forward(self, x, board=None):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 256*8*8)
        logits = self.fc(x)
        
        if board:
            evaluator = EvaluationEngine(board, depth=3)
            lookahead_score = evaluator.evaluate_position_with_lookahead()
            logits += lookahead_score * self.config.get('training', {}).get('rule_weight', 0.3)
        return logits

    def rule_based_evaluation(self, board):
        evaluator = EvaluationEngine(board, depth=3)  # Add depth parameter
        return evaluator.evaluate_position_with_lookahead()  # Use lookahead method


