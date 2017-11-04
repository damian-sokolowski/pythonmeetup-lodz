import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, Flatten, MaxPooling2D, AveragePooling2D, GaussianNoise
import pickle

sample_maxx = 30.0
sample_maxy = 30.0

generate_clouds = 0
generate_stars = 0

def set_clouds(c):
    global generate_clouds
    generate_clouds = c

def set_stars(s):
    global generate_stars
    generate_stars = s

def generate_sample():
    global sample_maxx
    global sample_maxx
    global generate_clouds
    global generate_stars

    r = 12.0
    while True:
        dx = int((0.5 - np.random.random_sample()) * r)
        dy = int((0.5 - np.random.random_sample()) * r)

        x1 = sample_maxx / 2.0 - int(np.random.random_sample() * 10)
        y1 = sample_maxx / 2.0 - int(np.random.random_sample() * 10)

        x2 = x1 + dx * 1
        y2 = y1 + dy * 1

        x3 = x1 + dx * 2
        y3 = y1 + dy * 2

        x4 = x1 + dx * 3
        y4 = y1 + dy * 3

        if (abs(dx) + abs(dy) > 0) & (x4 > 1) & (y4 > 1) & (x4 < sample_maxx - 2) & (y4 < sample_maxy - 2):
            break

    image1 = np.zeros((int(sample_maxx), int(sample_maxy)))

    if generate_clouds > 0:
        for x_temp in range(int(sample_maxx)):
            for y_temp in range(int(sample_maxy)):
                image1[x_temp, y_temp] = 255 * np.random.random_sample()
        image1 = gaussian_filter(image1, 2)

    image2 = np.copy(image1)
    image3 = np.copy(image2)

    if generate_stars > 0:
        for i in range(25):
            x_temp = int(np.random.random_sample() * 30)
            y_temp = int(np.random.random_sample() * 30)
            c = 200 + int(np.random.random_sample() * 50)
            image1[x_temp, y_temp] = c
            image2[x_temp, y_temp] = c
            image3[x_temp, y_temp] = c

    c = 255
    image1[int(x1), int(y1)] = c
    image2[int(x2), int(y2)] = c
    image3[int(x3), int(y3)] = c
    image1 = image1 / 255.0
    image2 = image2 / 255.0
    image3 = image3 / 255.0
    return image1, image2, image3, x4, y4



def generate_samples(n_samples = 50):
    global sample_maxx
    global sample_maxx

    X_temp = []
    y_temp = []

    for i in range(n_samples):
        image1, image2, image3, x4, y4 = generate_sample()
        t = np.append(image1, image2)
        t = np.append(t, image3)
        X_temp.append(t)
        y_temp.append(x4 / sample_maxx)

    X_temp = np.reshape(X_temp, (n_samples, int(3 * sample_maxx * sample_maxy)))
    y_temp = np.reshape(y_temp, (n_samples, 1))

    return X_temp, y_temp


def show(history):
    image1, image2, image3, x4, y4 = generate_sample()

    ax = plt.subplot2grid((3, 4), (0, 0))
    ax.grid(False)
    ax.axis('off')
    ax.set_title("0.1s")
    ax.imshow(image1, cmap="gray")

    ax = plt.subplot2grid((3, 4), (0, 1))
    ax.grid(False)
    ax.axis('off')
    ax.set_title("0.2s")
    ax.imshow(image2, cmap="gray")

    ax = plt.subplot2grid((3, 4), (0, 2))
    ax.grid(False)
    ax.axis('off')
    ax.set_title("0.3s")
    ax.imshow(image3, cmap="gray")

    ax = plt.subplot2grid((3, 4), (0, 3))
    ax.grid(False)
    ax.axis('off')
    ax.set_title("0.4s, impact")
    temp = np.copy(image3)
    temp[int(x4), int(y4)] = 1
    temp = temp + image2 * 0.7
    temp = temp + image1 * 0.3
    ax.imshow(temp, cmap="gray")

    ax = plt.subplot2grid((3, 4), (1, 0), colspan = 4)
    ax.set_yscale("log")
    ax.plot(history['loss'], 'b', label='Training loss')
    ax.plot(history['val_loss'], 'r', label='Validation loss')
    ax.legend(loc='upper right', shadow=True)

    ax = plt.subplot2grid((3, 4), (2, 0), colspan=4)
    ax.set_ylim([0, 1])
    ax.plot(history['acc'], 'g', label='Accuracy')
    ax.legend(loc='upper right', shadow=True)

    plt.show()

def train_model(model, n_iter, X_train, y_train, X_test = 0, y_test = 0, filename = ''):
    hist = {}
    hist['loss'] = []
    hist['val_loss'] = []
    hist['acc'] = []

    do_we_have_cv = 0
    do_we_need_to_learn = 1
    if type(X_test) is not type(0):
        do_we_have_cv = 1

    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam')

    if len(filename) > 0:
        try:
            input = open(filename, 'rb')
            hist = pickle.load(input)
            input.close()
            do_we_need_to_learn = 0
        except:
            pass

    if do_we_need_to_learn:
        for i in range(n_iter):
            if not do_we_have_cv:
                temp = model.fit(X_train, y_train, epochs=1, verbose=0)
                hist['loss'].append(temp.history['loss'])
                if i % 3 == 0:
                    y_pred = model.predict(X_train)
                    predictions = [value[0] for value in y_pred]
                    counter_all = 0
                    counter_good = 0
                    for y1, y2 in zip(y_train, predictions):
                        counter_all = counter_all + 1
                        if abs(y1 - y2) < 0.1:
                            counter_good = counter_good + 1
                    hist['acc'].append(float(counter_good) / counter_all)
                    print "%03d training loss %0.4f, accuracy %0.2f" % (
                    i + 1, temp.history['loss'][0], float(counter_good) / counter_all)
                else:
                    print "%03d training loss %0.4f" % (i + 1, temp.history['loss'][0])


            else:
                temp = model.fit(X_train, y_train, epochs=1, verbose=0, validation_data=(X_test, y_test))
                hist['loss'].append(temp.history['loss'])
                hist['val_loss'].append(temp.history['val_loss'])
                if i % 3 == 0:
                    y_pred = model.predict(X_test)
                    predictions = [value[0] for value in y_pred]
                    counter_all = 0
                    counter_good = 0
                    for y1, y2 in zip(y_test, predictions):
                        counter_all = counter_all + 1
                        if abs(y1 - y2) < 0.1:
                            counter_good = counter_good + 1
                    hist['acc'].append(float(counter_good) / counter_all)
                    print "%03d training %0.4f, validation %0.4f, accuracy %0.2f" % (
                        i + 1, temp.history['loss'][0], temp.history['val_loss'][0],
                        float(counter_good) / counter_all)
                else:
                    print "%03d training %0.4f, validation %0.4f" % (
                    i + 1, temp.history['loss'][0], temp.history['val_loss'][0])

        if len(filename) > 0:
            output = open(filename, 'wb')
            pickle.dump(hist, output)
            output.close()

    show(hist)
