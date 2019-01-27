import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import keras
from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras import regularizers
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split

#% matplotlib inline


X = np.load("npy_dataset/X.npy")
Y = np.load("npy_dataset/Y.npy")

print("shape of X:",X.shape)
print("shape of Y:",Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10, random_state=1)

x_train = x_train.reshape(-1,100,100,3)
x_test = x_test.reshape(-1,100,100,3)


#Our Model
def LeNet(width, height, channels, output):
    model = Sequential()
    
    #Convulation 1
    model.add(Conv2D(filters=32, kernel_size=(3,3)
                ,strides=(2,2)
                ,input_shape=(width, height, channels)))
    
    #ReLU Activation
    model.add(Activation('relu'))
    
    #Pooling
    model.add(MaxPool2D(pool_size=(2,2)))
    
    #Convolution 2
    model.add(Conv2D(filters=64, kernel_size=(3,3)
                    ,strides=(1,1)
                    ,kernel_regularizer=regularizers.l2(0.01)))
    
    #ReLU Activation
    model.add(Activation('relu'))
    
    
    #Pooling
    model.add(MaxPool2D(pool_size=(2,2)))
    
    #Convulation 3
    model.add(Conv2D(filters=100, kernel_size=(2,2)
                ,strides=(1,1)))
    
    #ReLU Activation
    model.add(Activation('relu'))
    
    #Pooling
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Flatten())
    
    #Hidden Layer
    model.add(Dense(100))
    
    model.add(Dropout(0.2))
    
    model.add(Activation('relu'))
    
    #Hidden Layer
    model.add(Dense(50))
    
    model.add(Dropout(0.2))
    
    model.add(Activation('relu'))
    
    model.add(Dense(output))
    
    model.add(Activation('softmax'))
    
    return model


def train():
	EPOCHS = 100

	model = LeNet(x_train.shape[1],x_train.shape[2],3,y_train.shape[1])

	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	t1 = time.time()
    
	hist_model =model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS)
	t2 = time.time()
	print("time required to train model ",t2-t1,"sec")
	loss = hist_model.history['loss']

	plt.plot(loss)



	model.save("model_rgb_opt.h5")


if __name__ == '__main__':
    train()