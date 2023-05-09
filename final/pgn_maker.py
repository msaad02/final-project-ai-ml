import chess
import chess.pgn
import collections

def board_to_game(board, event, white, black):
    game = chess.pgn.Game()

    # Undo all moves.
    switchyard = collections.deque()
    while board.move_stack:
        switchyard.append(board.pop())

    game.setup(board)
    node = game

    # Replay all moves.
    while switchyard:
        move = switchyard.pop()
        node = node.add_variation(move)
        board.push(move)

    game.headers["Result"] = board.result()
    game.headers["Event"] = event
    game.headers["White"] = white
    game.headers["Black"] = black
    return game