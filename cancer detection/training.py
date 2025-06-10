import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

PSA = np.array([3.8,3.4,2.9,2.8,2.7,2.1,1.6,2.5,2.0,1.7,1.4,1.2,0.9,0.8])
Group = np.array(['C','C','C','C','C','C','C','H','H','H','H','H','H','H'])
target = np.where(Group =='C',1,0) # if Group =='C' 'C' = 1 otherwise 'H' = 0

random.seed(915)
model = Sequential() # intialize the neural network
model.add(Input(shape=(1,)))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.1),loss='binary_crossentropy') # generate weights and biases for inputs
model.get_weights() # [weights,biases]

model.fit(PSA,target,epochs=1000,verbose=1)

input = np.array([2.5])
model.predict(input,verbose=0)