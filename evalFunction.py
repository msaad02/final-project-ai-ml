import numpy as np
import tensorflow as tf

def train_neural_network(bit_vector):
    # Define the neural network architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(13, 8, 8)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='rmsprop',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    # Prepare the training data
    x_train = np.array([bit_vector])
    y_train = np.array([0])  # Replace 0 with the actual label of the position

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=1)
    x_test = np.array([bit_vector])
    y_test = np.array([0])  # Replace 0 with the actual label of the test position

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(x_test, y_test)

    # Print the evaluation metrics
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

# train_neural_network(fen_to_bit_vector(fen))