import chess

def evaluate(board):
    if board.piece_at(chess.B3) != None:
        return 10
    if board.piece_at(chess.C6) != None:
        return -10
    return 0

def find_best_move(board, depth, maximizing_player):
    best_depth = float('-inf')
    if maximizing_player:
        best_score = float('-inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth - 1, float('-inf'), float('inf'), False)
            #print(score, best_score, best_move)
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
            print(score, best_score, best_move)
            if score < best_score:
                best_score = score
                best_move = move
            board.pop()
    
    return best_move
    
def minimax(board, depth, alpha, beta, maximizing_player):
    if board.is_game_over() or depth == 0:
        return evaluate(board)
    
    if maximizing_player:
        best_score = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth-1, alpha, beta, False)
            #print(move, score, depth)
            score += depth
            board.pop()
            best_score = max(score, best_score)
            #print(f'best {best_score}')
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_score
    else:
        best_score = float('inf')
        #print(board.turn)
        #print(board.legal_moves)
        print(board.legal_moves)
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth-1, alpha, beta, True)
            print(move, score, depth)
            score -= depth
            board.pop()
            #print(score, best_score)
            best_score = min(score, best_score)
            #print(f'best {best_score}')
            beta = min(beta, best_score)
            if alpha >= beta:
                break
        return best_score

board = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e3 0 1')
print(board.turn)
board.push(chess.Move.from_uci('d2d3'))
for i in range(1):
    print(board.turn)
    best_move = find_best_move(board, 2, False)
    
    print(best_move, '\n')
    board.push(best_move)
    print(board.turn)
    print(board, '\n')