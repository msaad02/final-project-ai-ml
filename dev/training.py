from tensorflow.keras.activations import sigmoid
import tensorflow as tf

from sequenceTest import DataSequence

def scaled_sigmoid(x):
    #print(2  * sigmoid(x) - 1)
    return 2  * sigmoid(x) - 1


# Define the neural network architecture
model = tf.keras.Sequential([
    #tf.keras.layers.Flatten(input_shape=(13, 8, 8)),
    tf.keras.layers.Dense(832, activation='linear'),
    tf.keras.layers.Dense(832, activation='relu'),
    tf.keras.layers.Dense(832, activation='linear'),
    tf.keras.layers.Dense(832, activation='relu'),
    tf.keras.layers.Dense(832, activation='linear'),
    tf.keras.layers.Dense(832, activation='relu'),
    #tf.keras.layers.Dense(1, activation=scaled_sigmoid)
    tf.keras.layers.Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam',
              loss='MeanAbsoluteError')

train_sequence = DataSequence('KaggleDataset/chessData.csv', 1024)

# Train the model
model.fit(train_sequence, epochs=10)

model.save('saved_model')