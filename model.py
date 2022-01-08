import tensorflow as tf
import os
import pickle
import numpy as np
from helpers import *
mnist = tf.keras.datasets.mnist

# Load data
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# Describe model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model 
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
# dump model definition 
pickle.dump(model, open('data/model', 'wb'))
# Set num_clients
num_clients = 10
# Set epochs 
epochs = 5

# Split data equally for each clients, where the indices specify the client - e.g. x_train[0] is client 0's data
# ! Note - must be able to be equally split . . . RaggedTensors beware.
x_train, y_train = split_data(x_train, num_clients), split_data(y_train, num_clients)

# Spin up containers and give them data
containers_up(x_train, y_train, num_clients)

for epoch in epochs:
    if epoch == 0:
        do_epoch_zero_things(model, x_train, y_train, num_clients)
        continue
    else: 
        # Average the weights from zeroth epoch results 
        avg_weights = avg_weights(num_clients)
        # Dump weights to a shared resource location - in this case, a file called 'weights' in the 'data' shared volume dir
        pickle.dump(avg_weights, open('data/weights', 'wb'))
    for client in range(num_clients):
        # spin up a docker container for it 





# # 

# # Fit data based on client1 or client2
# if client1: 
#     model.fit(x_train_first_half, y_train_first_half, epochs=5) # Training for all epochs 
#     pickle.dump(model, open('client_results/client1_results', 'wb'))
# elif client2: 
#     model.fit(x_train_second_half, y_train_second_half, epochs=5) # Training for all epochs 
#     pickle.dump(model, open('client_results/client2_results', 'wb'))

# # If not run from a docker container, run the evaluation after loading the data from the pickled files in ./client_results 
# else: 
#     # Load data from client_results dir 
#     model_1_results = pickle.load(open('client_results/client1_results', 'rb'))
#     model_2_results = pickle.load(open('client_results/client2_results', 'rb'))
#     # Shapes of weight arrays should be equal 
#     assert([weight_or_bias.shape for weight_or_bias in model_1_results.get_weights()] == [weight_or_bias.shape for weight_or_bias in model_2_results.get_weights()]) 
#     weight_arr = list()
#     # Average the weights
#     for wl1, wl2 in zip(model_1_results.get_weights(), model_2_results.get_weights()):
#         avg_weight = (wl1+wl2)/2
#         weight_arr.append(avg_weight) 
#     # Set weights    
#     model.set_weights(weight_arr)
#     # Evaluate model 
#     model.evaluate(x_test, y_test)
