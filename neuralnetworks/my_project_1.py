from utils import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

np.random.seed(2)

X, y = generate_samples(100)

model = Sequential()
model.add(Dense(10, input_dim = 3 * 30 * 30))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

train_model(model, 20, X, y)