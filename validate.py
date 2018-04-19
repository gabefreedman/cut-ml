from __future__ import print_function
from keras.models import load_model
from utils.get_cut import CutResults
import numpy as np
import keras
np.random.seed(100)

n_samples = 5000
n_tods = 10
# Retrieve data from api
cr = CutResults("/home/yguan/data/mr3_pa2_s16_results.pickle")
x_train, y_train, x_test, y_test = cr.get_data_learning(n_tods, n_samples, downsample=40)

# Data transformation
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
input_shape = (1, x_train.shape[1], 1)

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
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

model = load_model("100_model.h5")

# since we are validating, combine tests and train
x_test = np.vstack([x_train, x_test])
y_test = np.vstack([y_train, y_test])

prediction = model.predict(np.array(x_test))
print(np.hstack([prediction, y_test]))

# check the number of accurate guesses
correct = ((prediction[:,1] > prediction[:,0]) & (y_test[:,1] > y_test[:,0])) | ((prediction[:,1] < prediction[:,0]) & (y_test[:,1] < y_test[:, 0]))
print('Number of correct:', np.count_nonzero(correct))
print('Total:', prediction.shape[0])
print('Accuracy:', float(np.count_nonzero(correct)*100.0)/prediction.shape[0])
