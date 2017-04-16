## Training model
##
## import some useful python modules
import os
from pathlib import Path
import numpy as np
from numpy.random import random
import cv2
import pandas as pd
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


## model input files for initial model save and retrain
fileDataPath = './IMG'
fileDataCSV = '/driving_log.csv'

## model parameters defined here
ch, img_rows, img_cols = 3, 160, 320  # camera format
#ch, img_rows, img_cols = 3, 66, 200  # Nvidia's camera format
nb_classes = 1

## get our training and validation data
## features: center,left,right,steering,throttle,brake,speed
#print("\n\ntraining data from: ", fileDataPath+fileDataCSV)
data = pd.read_csv(fileDataPath+fileDataCSV)
#print( len(data), "read from: ", fileDataPath+fileDataCSV)
#print(data.describe())
#print("\n\ntypes:")
#print(data.dtypes)

data = data[(data.brake==0.0)&(data.speed>0.0)]
X_train = np.copy(data['center']+':'+data['left']+':'+data['right'])
Y_train = np.copy(data['steering'])
Y_train = Y_train.astype(np.float32)

## split the training data into training and validation
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=10)

## set up our hyper-parameters
batch_size = 20
samples_per_epoch = len(X_train)/batch_size
val_size = int(samples_per_epoch/10.0)
nb_epoch = 1000

# preprocess the images to what is in Nvidia's paper and crop the top 1/3.
def load_image(imagepath):
    imagepath = imagepath.replace(' ','')
    image = cv2.imread(imagepath, 1)
    # get shape and chop off 1/3 from the top
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[int(shape[0]/3):shape[0], 0:shape[1]]
    image = cv2.resize(image, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

def augment_images(X,Y):
    augmented_images = []
    augmented_measurements = []
    for i in range(len(X)):
        #image = load_image(X[i].split(':'))
        measurement=Y[i]
 	if measurement < -0.01:
		imagepath = X[i].split(':')[1]
		measurement *= 3.0
	else:
		if measurement > 0.01:
			imagepath = X[i].split(':')[2]
			measurement *= 3.0
		else:
			imagepath = X[i].split(':')[0]
	image = load_image(imagepath)
	augmented_images.appended(image)
	augmented_measurements.appended(measurement)
    flipped_image = cv2.flip(image, 1)
	flipped_measurement = float(measurement)*-1.0
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)
    yield augmented_images, augmented_measurements

input_shape = (img_rows, img_cols, ch)

# model architecture
pool_size = (2, 3)
model = Sequential()
model.add(MaxPooling2D(pool_size=pool_size,input_shape=input_shape))
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Convolution2D(5, 5, 24, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(5, 5, 36, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(5, 5, 48, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(3, 3, 64, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(3, 3, 64, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(1164))
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(100))
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(50))
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(10))
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(1))
adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])
history = model.fit_generator(augment_images(X_train, Y_train),
                    samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                    validation_data=(X_val, Y_val),
                    nb_val_samples=val_size,
                    verbose=1)
model.save('model.h5')
