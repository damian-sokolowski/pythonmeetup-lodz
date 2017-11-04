from utils import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv3D
import numpy as np

np.random.seed(7)
set_stars(1)
set_clouds(1)

X, y = generate_samples(3000)
X_valid, y_valid = generate_samples(300)

X = np.reshape(X, (X.shape[0], 30, 30, 3, 1))
X_valid = np.reshape(X_valid, (X_valid.shape[0], 30, 30, 3, 1))

model = Sequential()
model.add(Conv3D(filters=3, kernel_size=(3, 3, 3), input_shape=(30, 30, 3, 1), activation= 'sigmoid'))
model.add(Flatten())
model.add(Dense(3))
model.add(Dense(1))

train_model(model, 900, X, y, X_valid, y_valid, 'conv3d_900.pkl')
