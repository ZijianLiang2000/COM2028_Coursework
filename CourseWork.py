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
from matplotlib import image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
import os
# Print time for start execution of code
from tensorflow.python.keras.utils.np_utils import to_categorical


startTime = time.perf_counter()
print("Start execution time:", startTime)

# Initialize arrays
readLine = []
# create y_train to store y_train data
y_train = []
# store image pixel data for x_train and x_test
x_train1 = []
x_train2 = []
x_train3 = []
x_test = []

print("GPU number available:", len(tf.config.experimental.list_physical_devices("GPU")))


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

    scale_percent = 10  # percent of original size

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
x_train_image_range_from1 = 0
#  range(i, j) produces i, i+1, i+2, ..., j-1.
# So if range(0,500), meaning index [0 - 499], selects image 0 - 500
x_train_image_range_to1 = 1000
# So the next array starts from next index after previous array's last element
x_train_image_range_from2 = 1000
x_train_image_range_to2 = 3000
x_train_image_range_from3 = 3000
x_train_image_range_to3 = 5000
x_test_image_range_from = 0
x_test_image_range_to = 2

print("Processing array x_train to store images")
resizeImagesAndSave(x_train_images, x_train1, x_train_image_range_from1, x_train_image_range_to1)
resizeImagesAndSave(x_train_images, x_train2, x_train_image_range_from2, x_train_image_range_to2)
resizeImagesAndSave(x_train_images, x_train3, x_train_image_range_from3, x_train_image_range_to3)
print("Processing array x_test to store images")
resizeImagesAndSave(x_test_images, x_test, x_test_image_range_from, x_test_image_range_to)

# They needed to be normalized within 255
print("Normalising x_train")
x_train1 = np.array(x_train1) / 255
x_train2 = np.array(x_train2) / 255
x_train3 = np.array(x_train3) / 255
print("Normalising x_test")
x_test = np.array(x_test) / 255
print("Done")

# Convert x_train and x_test into greyscale img with shape (numIMG,400,600)
# print("Converting x_train into grayscale")
# x_train1 = rgb2gray(np.array(x_train1))
# x_train2 = rgb2gray(np.array(x_train2))
# x_train3 = rgb2gray(np.array(x_train3))
# print("Converting x_test into grayscale")
x_test = rgb2gray(np.array(x_test))

# Reshape x_train and x_test images into -1,400,600,1
# Now shape of x_train is (500, 400, 600)
# shape of x_test is(300, 400, 600)
# print("Reshaping x_train1")
# x_train1 = np.array(x_train1).reshape(-1, 400, 600, 1)
# print("Reshaping x_train2")
# x_train2 = np.array(x_train2).reshape(-1, 400, 600, 1)
# print("Reshaping x_train3")
# x_train3 = np.array(x_train3).reshape(-1, 400, 600, 1)
# print("Reshaping x_test")
# x_test = np.array(x_test).reshape(-1, 400, 600, 1)

# Turn y_train to be categorical
# y_train will be 1 value initially, either class 0,1,2 ... 22 with shape (10270,)
# categorize y_train into (10270,23), which shows the most possible class
print("y_train being processed to be categorical")
y_train = to_categorical(y_train, 23)
print("Categorical process finished")

print("Processing y_train to constraint range")
y_train = y_train[x_train_image_range_from1:x_train_image_range_to3]
print("Finished processing y_train to constraint range")


# Referenced from https://www.geeksforgeeks.org/how-to-load-and-save-3d-numpy-array-to-file-using-savetxt-and-loadtxt
# -functions/ reshaping the array from 4D matrices to 2D matrices to store into csv

# print("Processing x_train reshape to save into csv form")
# x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
# print("Finished processing x_train reshape to save into csv form")
# print("Processing x_test reshape to save into csv form")
# x_test_reshaped = x_test.reshape(x_train.shape[0], -1)
# print("Finished processing x_test reshape to save into csv form")
# print("Saving x_train into csv")
# numpy.savetxt("x_train.csv", x_train_reshaped, delimiter=",")
# print("x_train saved into csv")
# print("Saving x_test into csv")
# numpy.savetxt("x_test.csv", x_test_reshaped, delimiter=",")
# print("x_test saved into csv")

def loadArrayFromCsv(arrayNameToLoad, originalArray):
    # retrieving data from file.
    loaded_arr = numpy.loadtxt(arrayNameToLoad)

    # This loadedArr is a 2D array, therefore
    # we need to convert it to the original
    # array shape.reshaping to get original
    # matrice with original shape.
    load_original_arr = loaded_arr.reshape(loaded_arr.shape[0], loaded_arr.shape[1] // originalArray.shape[2])

    # check the shapes:
    print("shape of arr: ", originalArray.shape)
    print("shape of load_original_arr: ", load_original_arr.shape)

    # check if both arrays are same or not:
    if (load_original_arr == originalArray).all():
        print("Yes, both the arrays are same")
    else:
        print("No, both the arrays are not same")

    return load_original_arr


# Concatenating 2 x_train arrays
x_train = np.concatenate((x_train1, x_train2, x_train3))

# Keras Model Part
print("Building model...")

model = Sequential()

# 1st model
# # Model Conventional2D kernel should be a batch size of 64, with input shape of 400*600*1
model.add(Conv2D(32, kernel_size=(5, 5), activation="relu", input_shape=(40, 60, 3), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(5, 5), strides=2))
model.add(Conv2D(64, kernel_size=(5, 5), activation="relu", strides=(1, 1), padding='valid'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=2))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
# size of output layer should be 3 classes
model.add(Dense(23, activation="softmax"))
print("Training and fitting model...")
optimizer = keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

modelTrainStart = time.perf_counter()
model.fit(x_train, y_train, validation_split=0.3, epochs=50, batch_size=8, shuffle=True)
# Epochs around 20 gives best loss and val_accuracy
model.save("my_model_butterfly")

# It can be used to reconstruct the model identically.
# reconstructed_model = keras.models.load_model("my_model")


# Print time for start execution of code
endTime = time.perf_counter()
print("End execution time of code:", endTime)
print("Total time duration of code:", endTime - startTime, "seconds processing from", x_train_image_range_from1, "to",
      x_train_image_range_to2,
      "images from x_train, the corresponding y_train and from", x_test_image_range_from, "to", x_test_image_range_to,
      "images from x_test.")
print("Model training time:", endTime-modelTrainStart,"seconds")