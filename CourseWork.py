import random
import cv2
import keras
import glob
import re
import time

import numpy
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import models, layers
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization
from keras.regularizers import l2
import os
# Print time for start execution of code
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

startTime = time.perf_counter()
print("Start execution time:", startTime)

# Initialize arrays
readLine = []
# create y_train to store y_train data
y_train = []
# store image pixel data for x_train and x_test
x_train = []
x_test = []

print("GPU number available:", len(tf.config.experimental.list_physical_devices("GPU")))

# def storeX_TrainWithLabels(x_trainArray, y_trainArray, x_validArray, y_validArray, trainPath, validPath):
#     print("Storing image and labels")
#     for length in range(len(x_trainArray)):
#         for y_range in range(23):
#             if int(y_trainArray[length]) == y_range:
#                 img = Image.fromarray(x_trainArray[length])
#                 createLabelFolders(str(trainPath) + str(y_range))
#                 if os.path.isfile(str(trainPath) + str(y_range) + "/" + str(length) + ".jpg"):
#                     print("File already exists, jump to next file")
#                 else:
#                     img.save(str(trainPath) + str(
#                         y_range) + "/" + str(length) + ".jpg")
#     for length in range(len(x_validArray)):
#         for y_range in range(23):
#             if int(y_validArray[length]) == y_range:
#                 img = Image.fromarray(x_validArray[length])
#                 createLabelFolders(str(validPath) + str(y_range))
#                 if os.path.isfile(str(validPath) + str(y_range) + "/" + str(length) + ".jpg"):
#                     print("File already exists, jump to next file")
#                 else:
#                     img.save(str(trainPath) + str(
#                         y_range) + "/" + str(length) + ".jpg")
#     print("Finished")


# def createLabelFolders(path):
#     try:
#         os.mkdir(path)
#     except OSError:
#         print("Directory already created", path)
#     else:
#         print("Successfully created the directory %s ", path)


def displayImage(singleImage):
    img = Image.fromarray(singleImage)
    img.show()


# Upload file "train.txt" containing y_train data
# uploaded = files.upload()
def openY_Train(file):
    uploaded = open(file, "r")
    for everyLine in uploaded:
        readLine.append(everyLine)


def loadY_Train():
    # Put only the y_train data into array in order
    for line in readLine:
        # replace unnecessary strings with space
        temp = line.replace("\n", "")
        splitChar = " "
        # Partition out only the numbers and add to array
        temp1 = temp.partition(splitChar)[2]
        y_train.append(temp1)


# Refernce from https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# Load x_train images in local directory
x_train_images = glob.glob(
    "E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/train/train" + '/*.jpg')
x_train_images.sort(key=natural_keys)

# Load x_test images in local directory
x_test_images = glob.glob("E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/test/test" + '/*.jpg')
x_test_images.sort(key=natural_keys)


# Resize all images into unified size, since some photo have different size
# Resize a single image into shape 400*600*3, meaning 400px * 600px * 3 channels (RGB)
def resizeImagesAndSave(imgArray, arrayToSave, imageRangeFrom, imageRangeTo):
    # for singleImage in range(len(images)):
    print("Processing array to store images")

    scale_percent = 8  # percent of original size

    for singleImage in range(imageRangeFrom, imageRangeTo):
        print("Currently processing the", singleImage, "image")

        # All images should be cropped to this size
        cropImage = np.resize(image.imread(imgArray[singleImage]), (400, 600, 3))

        width = int(cropImage.shape[1] * scale_percent / 100)
        height = int(cropImage.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(cropImage, dim, interpolation=cv2.INTER_AREA)
        # Saved to specific array parameter
        arrayToSave.append(resized)
        # print("Image shape after crop:", np.array(cropImage).size)
    print("Process finished for array to store images")


# Function to convert 3 channel rgb image to greyscale 1 channel Referenced from:
# https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python by Mark Amery
def rgb2gray(rgb):
    print("Grey-Scaling images for array")
    r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    print("Grey-Scaling process finished for array")
    return gray


# Run to open and load y_train in local terminal
openY_Train("train.txt")
# y_train is ranged from 0 to 22
loadY_Train()

# Resize x_train and x_test images into unified size and save into array, with range of images constrained
# x_train_image_range should be max 10270
# x_test_image_range should be max 15009
# Start from index 0
x_train_image_range_from = 0
#  range(i, j) produces i, i+1, i+2, ..., j-1.
# So if range(0,500), meaning index [0 - 499], selects image 0 - 500
x_train_image_range_to = 10270
x_test_image_range_from = 0
x_test_image_range_to = 6

print("Processing array x_train to store images")
resizeImagesAndSave(x_train_images, x_train, x_train_image_range_from, x_train_image_range_to)
print("Processing array x_test to store images")
resizeImagesAndSave(x_test_images, x_test, x_test_image_range_from, x_test_image_range_to)

image = Image.fromarray(x_train[0])
image.show()

# X_train and x_validation spreaded as 8216 and 2054
# X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=13)
#
# storeX_TrainWithLabels(X_train,y_train, X_val, y_val,"E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/data_split/train/", "E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/data_split/validation/")
#
# train_datagen = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)
# validation_datagen = ImageDataGenerator(rescale=1. / 255)
#
# BATCH_SIZE = 16
# IMAGE_SIZE = (48, 32)
#
# train_generator = train_datagen.flow_from_directory(
#     'E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/data_split/train',
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='binary')
# validation_generator = validation_datagen.flow_from_directory(
#     'E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/data_split/validation',
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='binary')

# They needed to be normalized within 255
print("Normalising x_train")
x_train = np.array(x_train) / 255
print("Normalising x_test")
x_test = np.array(x_test) / 255
print("Done")

# Convert x_train and x_test into greyscale img with shape (numIMG,400,600)
# print("Converting x_train into grayscale")
# x_train = rgb2gray(np.array(x_train))
# print("Converting x_test into grayscale")
# x_test = rgb2gray(np.array(x_test))

# Reshape x_train and x_test images into -1,400,600,1
# Now shape of x_train is (500, 400, 600)
# shape of x_test is(300, 400, 600)
print("Reshaping x_train1")
x_train = np.array(x_train).reshape(-1, 32, 48, 3)
print("Reshaping x_test")
x_test = np.array(x_test).reshape(-1, 32, 48, 3)

print(np.array(x_train).shape)

# Turn y_train to be categorical
# y_train will be 1 value initially, either class 0,1,2 ... 22 with shape (10270,)
# categorize y_train into (10270,23), which shows the most possible class
print("y_train being processed to be categorical")
y_train = to_categorical(y_train, 23)
print("Categorical process finished")
#
print("Processing y_train to constraint range")
y_train = y_train[x_train_image_range_from:x_train_image_range_to]
print("Finished processing y_train to constraint range")

# Keras Model Part
print("Building model...")

model = Sequential()

# print("X_train model size", np.array(X_train).shape)
# print("X_valid model size",np.array(X_val).shape)

# 1st model
# # Model Conventional2D kernel should be a batch size of 64, with input shape of 400*600*1
# model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=(32, 48, 3), strides=(1, 1)))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", strides=(1, 1), padding='valid'))
# model.add(MaxPooling2D(pool_size=(2,2)))
#
# model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", strides=(1, 1), padding='valid'))
# model.add(MaxPooling2D(pool_size=(2,2)))
#
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation("relu"))

# 2nd model
input_shape=(32, 48, 3)
cnn4 = Sequential()
cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.01)))
cnn4.add(BatchNormalization())

cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
cnn4.add(Dropout(0.25))

cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.25))

cnn4.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
cnn4.add(Dropout(0.25))

cnn4.add(Flatten())

cnn4.add(Dense(512, activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.5))

cnn4.add(Dense(128, activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.5))

# size of output layer should be 3 classes
cnn4.add(Dense(23, activation="softmax"))

opt = keras.optimizers.Adam(learning_rate=0.01)
cnn4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn4.fit(x_train,y_train, validation_split=0.3, epochs=25, batch_size=128, shuffle=True)
model.save("my_model_butterfly_cnn4")
# Third model (lower accuracy)
# cnn = Sequential()
# cnn.add(Conv2D(filters=32,
#                kernel_size=(2, 2),
#                strides=(1, 1),
#                padding='same',
#                input_shape=(32, 48, 3),
#                data_format='channels_last'))
# cnn.add(Activation('relu'))
# cnn.add(MaxPooling2D(pool_size=(2, 2),
#                      strides=2))
# cnn.add(Conv2D(filters=64,
#                kernel_size=(2, 2),
#                strides=(1, 1),
#                padding='valid'))
# cnn.add(Activation('relu'))
# cnn.add(MaxPooling2D(pool_size=(2, 2),
#                      strides=2))
# cnn.add(Flatten())
# cnn.add(Dense(64))
# cnn.add(Activation('relu'))
# cnn.add(Dropout(0.25))
# cnn.add(Dense(1))
# cnn.add(Activation('sigmoid'))
# cnn.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#
# print("Training and fitting model...")

# start = time.time()
# cnn.fit_generator(
#     train_generator,
#     steps_per_epoch=8216 // BATCH_SIZE,
#     epochs=50,
#     validation_data=validation_generator,
#     validation_steps=2054 // BATCH_SIZE)
# end = time.time()
# print('Processing time:', (end - start) / 60)
# cnn.save_weights('cnn_baseline.h5')
# It can be used to reconstruct the model identically.
# reconstructed_model = keras.models.load_model("my_model")


# Print time for start execution of code
# print("End execution time of code:", end)
# print("Total time duration of code:", end - start, "seconds processing from", x_train_image_range_from, "to",
#       x_train_image_range_to,
#       "images from x_train, the corresponding y_train and from", x_test_image_range_from, "to", x_test_image_range_to,
#       "images from x_test.")
# # print("Model training time:", endTime-modelTrainStart,"seconds")
