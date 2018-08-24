from __future__ import print_function

from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalMaxPooling2D
from keras.utils import np_utils
import numpy as np
from utils.get_cut import *

np.random.seed(10)
batch_size = 32 
num_classes = 2
epochs = 10
n_samples = 2000  # number of positive samples in dataset -> adjust
                  # based on n_tods
n_tods = 5
downsample = 40
n_repeat = 1 # number of repeats
input_shape = (1, None, 1) # test varaible input size

# Retrieve data from api
cr = CutResults("mr3_pa2_s16_results.pickle")
def generate_dataset(n_tods, n_samples, downsample):
    x_train, y_train, x_test, y_test = cr.get_data_learning(n_tods,
                                                            n_samples, downsample=downsample)
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))
    unique, counts = np.unique(y_test, return_counts=True)
    print(dict(zip(unique, counts)))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #input_shape = (1, x_train.shape[1], 1)

    # Conform to Conv2D input requirement last digit is channel
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], 1)


    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # output labels should be one-hot vectors - ie,
    # 0 -> [0, 1]
    # 1 -> [1, 0]

    # this operation changes the shape of y from (10000,1) to (10000, 3)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
   
    return x_train, y_train, x_test, y_test

# define a CNN
# see http://keras.io for API reference
model = Sequential()
model.add(Conv2D(256, kernel_size=(1, 10), strides=(1, 2),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
model.add(Conv2D(128, (1, 10), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 2)))

model.add(Conv2D(128, (1, 10), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 2)))
#model.add(Flatten())  # useful when shape is not (1, None, 1)
model.add(GlobalMaxPooling2D()) # try to fix the flatten problem with GlobalMaxPooling2D
model.add(Dropout(0.25))

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

for i in range(n_repeat):
    print("============ ITERATION %d ==============" % i)
    x_train, y_train, x_test, y_test = generate_dataset(n_tods, n_samples, downsample)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
prediction = model.predict(np.array(x_test))
print(np.hstack([prediction, y_test]))
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# model.save('my_model.h5')
plt.plot(range(1, epochs+1), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
