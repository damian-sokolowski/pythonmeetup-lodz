from utils import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

np.random.seed(2)
set_stars(0)
set_clouds(0)

X, y = generate_samples(2500)
X_valid, y_valid = generate_samples(300)

model = Sequential()
model.add(Dense(5, input_dim=3 * 30 * 30))
model.add(Dropout(0.25))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))

train_model(model, 100, X, y, X_valid, y_valid)
