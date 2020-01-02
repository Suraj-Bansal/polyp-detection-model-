#MACHINE LEARNING BASED POLYP DETECTION 
#SURAJ BANSAL- DECEMBER, 13 2019 

# Importing neccesary libraries
import tensorflow as tf 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import keras
import itertools

from matplotlib import cm

from keras import layers
from keras import models
from keras import datasets 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.layers. normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# Declare some useful values for later 

#number of training examples for each dataset
num_train_samples = 1224
num_val_samples = 1219

#number of images the model tests and trains with for each epoch
train_batch_size = 20
val_batch_size = 20

#re-sizing images 
image_size = 224

# Declaring number of steps needed for each iteration
train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

#Creating paths for training and validation dataset 
train_path = './DATA/train'
valid_path = './DATA/valid'

#Insert random image generators for the training, validation and testing sets  
train_batches = ImageDataGenerator(
    preprocessing_function= \
        keras.applications.mobilenet.preprocess_input).flow_from_directory(
    train_path,
    target_size=(image_size, image_size),
    batch_size=train_batch_size,
    class_mode= 'binary'
    )

valid_batches = ImageDataGenerator(
    preprocessing_function= \
        keras.applications.mobilenet.preprocess_input).flow_from_directory(
    valid_path,
    target_size=(image_size, image_size),
    batch_size=val_batch_size,
    class_mode= 'binary'
    )

test_batches = ImageDataGenerator(
    preprocessing_function= \
        keras.applications.mobilenet.preprocess_input).flow_from_directory(
    valid_path,
    target_size=(image_size, image_size),
    batch_size=val_batch_size,
    shuffle=False,
    class_mode= 'binary'
    )

#Train the model
model = models.Sequential()

#adding convolution layers to extract features
#using ReLU activation to introduce nonlinearity
#adding max pooling layers to reduce spatial parameters 
#introducing dropout layers to mitigate chances of overfitting
#using batch normalization to normalize the activations of each layer 
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape = (224,224,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',))
model.add(layers.MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',))
model.add(layers.MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',))
model.add(layers.MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',))
model.add(layers.MaxPooling2D((2, 2)))

#print summary 
model.summary()

#unrolling 3D output array into 1D vector
model.add(layers.Flatten())
#adding dense layers to reduce spatial parameters of vector
model.add(layers.Dense(64, activation = 'relu'))
model.add(Dropout(0.25))
#introducing softmax function to convert output into probability of each class being true
model.add(layers.Dense(2, activation = 'softmax'))

#printing model summary
model.summary()

#training the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics = ['accuracy']
    )

# Define Top2 and Top3 Accuracy
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

#declare the filepath for the saved model
filepath = "model.h5"

#declare checkpoints to save the best version of the model
checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1,
                             save_best_only=True, mode='max')

#reduce the learning rate as the learning stagnates
reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2,
                              verbose=1, mode='max', min_lr=0.00001)

callbacks_list = [checkpoint, reduce_lr]

#fit the model
#train for 50 epochs
history = model.fit_generator(train_batches,
                              steps_per_epoch=train_steps,
                              validation_data=valid_batches,
                              validation_steps=val_steps,
                              epochs=50,
                              verbose=1,
                              callbacks=callbacks_list)