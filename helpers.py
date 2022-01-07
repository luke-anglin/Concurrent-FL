import numpy as np
import pickle
def split_data(data, num_clients):
    '''
    Returns data divided for each num_client. 
    Client's 0 data accessed at zeroeth index of the returned list
    '''
    return np.split(data, num_clients)

def do_epoch_zero_things(model, x_train, y_train, num_clients):
    '''
    Does epoch zero separately to avoid a O(n^3) for loop.
    '''
    for client in range(num_clients):
        # Clients data is at idx (x_train[0] is client0's portion of the data)
        # Train for one epoch
        model.fit(x_train[client], y_train[client], epochs=1)
        # Write fitted model to client_results/client{CLIENT_#} - e.g. client_results/client0 for the zeroth client 
        pickle.dump(model, open(f'client_results/client{client}_results', 'wb'))

def avg_weights(num_clients):
    '''
    Loads pickled data from ./client_results directory
    Averages weights
    Returns the array of weights
    '''
    client_weights = list() # Weights for client 0 will be in 0th index
    # Load all data
    for client in range(num_clients): 
        model = pickle.load(model, open(f'client_results/client{client}_results', rb))
        client_data.append(model.get_weights()) 
    avg_weights = np.mean(a, axis=0) # See docs to understand why axis=0 at https://numpy.org/doc/stable/reference/generated/numpy.mean.html
    
    
        