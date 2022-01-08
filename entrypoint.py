import pickle
import tensorflow as tf
import time 

# get model def 
model = pickle.load(open('data/model', 'rb'))
prev_weights = list()

while (True):
    # Sleep if weights haven't been updated yet
    if (weights == prev_weights):
        time.sleep(2.5)
        continue
    # Weights have been updated, load in new average weights
    weights = pickle.load(open('client_results/weights', 'rb'))
    # Set the weights
    model.set_weights(weights)
    # Fit the model to our section of the data
    model.fit()
