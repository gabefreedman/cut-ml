from keras.models import load_model
from utils.get_cut import CutResults
import numpy as np
import keras

n_samples = 100
# Retrieve data from api
cr = CutResults("/home/yguan/data/mr3_pa2_s16_results.pickle")
x_train, y_train, x_test, y_test = cr.get_data_learning(1, n_samples, downsample=40)

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

model = load_model("my_model.h5")

prediction = model.predict(np.array(x_test))
print(np.hstack([prediction, y_test]))

# check the number of accurate guesses
if prediction[i][1]>prediction[i][0]:
