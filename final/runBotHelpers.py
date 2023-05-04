import chess
import utils
import numpy as np

def evaluate(board, model):
    evaluation = model.predict_step(np.asarray(utils.processFEN(board.fen())))
    return evaluation

def find_best_move(board, depth, maximizing_player, model):
    if maximizing_player:
        best_score = float('-inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth - 1, float('-inf'), float('inf'), False, model)
            print(move, score)
            if score > best_score:
                best_score = score
                best_move = move
            board.pop()
    else:
        best_score = float('inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth - 1, float('-inf'), float('inf'), True, model)
            if score < best_score:
                best_score = score
                best_move = move
            board.pop()
    
    return best_move

# def minimax(board, depth, alpha, beta, maximizing_player, model):
#     if board.is_game_over() or depth == 0:
#         return evaluate(board, model)

def minimax(board, depth, alpha, beta, maximizing_player, model):
    if board.is_checkmate():
        return -10000 if maximizing_player else 10000
    if board.is_game_over(claim_draw=True):
        return 9000 if maximizing_player else -9000
    if depth == 0:
        return evaluate(board)
    
    if maximizing_player:
        best_score = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth-1, alpha, beta, False, model)
            score += depth
            board.pop()
            best_score = max(score, best_score)
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_score
    else:
        best_score = float('inf')
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth-1, alpha, beta, True, model)
            score -= depth
            board.pop()
            best_score = min(score, best_score)
            beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_score