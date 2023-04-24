import tensorflow as tf
import chess

import utils

def evaluate(board):
    #print(vectorize(board.fen()).shape)
    evaluation = model.predict_step(utils.vectorize(board.fen()).reshape((1,832)))
    #print(evaluation)
    return round(float(evaluation), 4)

def find_best_move(board, depth, maximizing_player):
    if maximizing_player:
        best_score = float('-inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth - 1, float('-inf'), float('inf'), False)
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
            score = minimax(board, depth - 1, float('-inf'), float('inf'), True)
            if score < best_score:
                best_score = score
                best_move = move
            board.pop()
    
    return best_move
    
def minimax(board, depth, alpha, beta, maximizing_player):
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
            score = minimax(board, depth-1, alpha, beta, False)
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
            score = minimax(board, depth-1, alpha, beta, True)
            score -= depth
            board.pop()
            best_score = min(score, best_score)
            beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_score


if __name__ == '__main__':
    model = tf.keras.models.load_model('saved_model')
    
    board = chess.Board()
    depth = 3

    while not board.is_game_over():
        print(board)
        print()

        if board.turn == chess.WHITE:
            move = find_best_move(board, depth, True)
            print(move)
        else:
            move = chess.Move.from_uci(input(prompt='Move: '))
            

        board.push(move)

    print("Game over")
    print("Result:", board.result())