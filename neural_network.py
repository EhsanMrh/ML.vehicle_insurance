# Here i will implement neural network for vehicle insurance project

import keras
from keras.models import Sequential
from keras.layers import Dense

def neural_model(train_data):
    
    # Create model
    network = Sequential()
    
    # Input layer
    network.add(Dense(units = 15 , activation = 'relu', init = 'uniform', input_dim = train_data.shape()[1]))    
    
    # Seconde layer
    network.add(Dense(units = 30, activation = 'relu', init = 'uniform'))
    
    # Third layer 
    network.add(Dence(units = 10, activation = 'relu', init = 'uniform'))
    
    # Output layer
    network.add(Dence(units = 1, activation = 'sigmoid', init = 'uniform'))