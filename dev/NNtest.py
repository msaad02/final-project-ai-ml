import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Define the neural network architecture
model = tf.keras.Sequential([
    #tf.keras.layers.Flatten(input_shape=(13, 8, 8)),
    tf.keras.layers.Dense(832, activation='relu'),
    tf.keras.layers.Dense(400, activation='relu'),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='RMSprop',
              loss='MeanSquaredError',
              metrics=['accuracy'])

def train_neural_network(positions, scores):

    # Set training and testing data
    x_train, x_test, y_train, y_test = train_test_split(positions, scores, random_state=0, train_size = .75)
    print(x_train.shape)
    x_train = x_train.reshape(len(x_train), 832)
    x_test = x_test.reshape(len(x_test), 832)
    #print(x_train)
    print(x_train.shape)
    print(y_train)
    
    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=128)

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(x_test, y_test)

    # Print the evaluation metrics
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    
def vectorize(fen):
    data = re.split(" ", fen)
    rows= re.split("/", data[0])
    turn = data[1]
    can_castle = data[2]
    passant = data[3]
    half_moves = data[4]
    full_moves = data[5]
    
    bit_vector = np.zeros((13, 8, 8), dtype=np.float32)
    #print(bit_vector.shape)
    #what layer each piece is found on
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
    #find each piece based on type
    for r,value in enumerate(rows):
        colum = 0
        for piece in value:
            if piece in piece_to_layer:
                bit_vector[piece_to_layer[piece],r,colum] =1
                colum += 1
            else:
                colum += int(piece)
    
    if turn.lower() == 'w':
        bit_vector [0,7,4] =1
    else:
        bit_vector [0,0,4] =1
        
    #where each castle bit is located
    castle ={
        'k': (0,0),
        'q': (0,7),
        'K': (7,0),
        'Q': (7,7),
        }

    for value in can_castle:
        if value in castle:
            bit_vector[0,castle[value][0],castle[value][1]] = 1
    
    #put en-passant square in the vector
    if passant != '-':
        bit_vector[0,  5 if (int(passant[1])-1 == 3) else 2 , ord(passant[0]) - 97,] = 1
    
    print(bit_vector.size)
    return bit_vector

def preprocess_scores(scores):
    #sets all scores within -150 to 150
    #mate scores are set to +-151
    for i, score in enumerate(scores):
        if score[0:2] == '#+':
            scores[i] = 151
        elif score[0:2] == '#-':
            scores[i] = -151
        elif int(score) > 150:
            scores[i] = 150
        elif int(score) < -150:
            scores[i] = -150
        else:
            scores[i] = int(score)
    #scale between 0 and 1
    scores += 150
    scores /= 300
    scores = scores.astype('float32')
    return scores

df = pd.read_csv('KaggleDataset/chessData.csv', nrows=1000, dtype={'FEN':str, 'Evaluation':str})
positions = np.ndarray(shape=(1000,13,8,8))
for i, f in enumerate(df.FEN):
    positions[i] = vectorize(f)
scores = df.Evaluation.to_numpy()
scores = preprocess_scores(scores)

train_neural_network(positions, scores)