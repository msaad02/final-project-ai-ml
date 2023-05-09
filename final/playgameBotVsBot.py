import tensorflow as tf
import chess
import runBotHelpers
from pgn_maker import board_to_game

model1 = tf.keras.models.load_model('12M_25e_64bs')
model2 = tf.keras.models.load_model('12M')

board = chess.Board()
depth = 3

while not board.is_game_over():
    print(board)
    print()

    if board.turn == chess.WHITE:
        move = runBotHelpers.find_best_move(board, depth, True, model1)
        print(move)
    else:
        move = runBotHelpers.find_best_move(board, depth, False, model2)
        print(move)
        
    board.push(move)

event = '12M 3 epochs 64 batch size vs 10M 20 epochs 8192 batch size'
white = '12M_25e_64bs'
black = '12M'

print(board_to_game(board, event, white, black))

print("Game over")
print("Result:", board.result())