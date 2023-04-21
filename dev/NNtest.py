import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import vectorize
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