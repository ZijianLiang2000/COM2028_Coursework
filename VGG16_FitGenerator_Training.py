import os
import pickle

import cv2
import glob
import re
import time
from keras.applications import vgg16 as vgg16
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers as layers
from matplotlib import image
# Print time for start execution of code
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers as optimizers
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping

startTime = time.perf_counter()
print("Start execution time:", startTime)

# 10270 images for x_train
X_TRAIN_IMG_ARRAY_LENGTH = 10270

# Initialize arrays
readLine = []
# create y_train to store y_train data
y_train = []
# store image pixel data for x_train and x_test
x_train = []

# Standardize function for image arrays
standardScaler = StandardScaler()

print("GPU number available:", len(tf.config.experimental.list_physical_devices("GPU")))

# The following functions are for data augmentation and fit_generator
# storeX_TrainWithLabelsForX_Test(x_test,"E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/test1/")
#
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    # rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    # rescale=1. / 255
    )

BATCH_SIZE = 32
IMAGE_SIZE = (200, 200)

train_generator = train_datagen.flow_from_directory(
    'E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/data_split_new/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    'E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/data_split_new/validation',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')
print("Done")


# Keras Model Part
print("Building model...")

VGG16 = keras.applications.vgg16.VGG16(input_shape=(200, 200, 3), include_top=False, weights="imagenet", classes=23)
VGG16.trainable = False

model = keras.Sequential([
    VGG16,
    keras.layers.Flatten(),
    keras.layers.Dense(units=4096, activation="relu"),
    keras.layers.Dense(units=4096, activation="relu"),
    # keras.layers.Dropout(0.2),
    keras.layers.Dense(units=23, activation="softmax")
])

print("Predicting model")

# opt = optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

history = model.fit_generator(generator=train_generator, validation_data=validation_generator,
                              steps_per_epoch=8216 // BATCH_SIZE,
                              epochs=50,
                              validation_steps=2054 // BATCH_SIZE,
                              callbacks=[checkpoint,early])

# Current best is cnn1 - 76% actual accuracy
model_name = "my_model_butterfly_VGG16_FitGenerator_Latest"
print("Saving model")
model.save(str(model_name))
print("Model saved as", str(model_name))

# Plot accuracy and loss
# plt.plot(history.history["accuracy"])
# plt.plot(history.history["val_accuracy"])
# plt.title("model accuracy")
# plt.xlabel("epoch")
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

with open('/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
