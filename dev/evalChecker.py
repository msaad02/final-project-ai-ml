import tensorflow as tf
import chess
import utils


def checkAllMoves(board):
    for move in board.legal_moves:
        board.push(move)
        score = model.predict_step(utils.vectorize(board.fen()).reshape((1,832)))
        print(move, score)
        board.pop()
        
if __name__ == '__main__':
    print('loading model')
    model = tf.keras.models.load_model('saved_model')
    print('model loaded')
    fen = ''
    while fen != "exit":
        fen = input("enter fen string:")
        board = chess.Board(fen)
        checkAllMoves(board)
       