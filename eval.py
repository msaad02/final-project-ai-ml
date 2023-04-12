import re
import numpy as np
# what needs to be done, 
#vecorize the boardstate 
#start by finding all the pices and show where they are on the board
# fen exmpale:
# given a fen string
def vectorize(fen):
    data = re.split(" ", fen)
    rows= re.split("/", data[0])
    
    bit_vector = np.zeros((13, 8, 8), dtype=np.uint8)
    piece_to_layer = {
            'R': 1,
            'N': 2,
            'B': 3,
            'Q': 4,
            'K': 5,
            'P': 6,
            'p': 7,
            'k': 8,
            'q': 9,
            'b': 10,
            'n': 11,
            'r': 12
        }
    for r,value in enumerate(rows):
        colum = 0
        for piece in value:
            if piece in piece_to_layer:
                bit_vector[piece_to_layer[piece],r,colum] =1
                colum += 1
            else:
                colum += int(piece)
    return bit_vector
    
print(vectorize(r'rnbqkb1r/ppp1pppp/3p4/3nP3/3P4/8/PPPK1PPP/RNBQ1BNR b kq - 1 4'))