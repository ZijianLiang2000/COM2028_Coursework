import random
import cv2
import pandas
from keras.models import load_model
from tensorflow import keras
import glob
import re
import time
from keras.applications import vgg16 as vgg16
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import models, layers, applications
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import image
# Print time for start execution of code
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

startTime = time.perf_counter()
print("Start execution time:", startTime)

X_TRAIN_IMG_ARRAY_LENGTH = 10270

X_TEST_IMG_ARRAY_LENGTH = 15009

# Initialize arrays
readLine = []
# create y_train to store y_train data
y_train = []
# store image pixel data for x_train and x_test
x_train = []
x_test = []

# Standardize function for image arrays
scaler = StandardScaler()

x_train = np.empty((X_TRAIN_IMG_ARRAY_LENGTH, 200, 200, 3), dtype=np.float32)
x_test = np.empty((X_TEST_IMG_ARRAY_LENGTH, 200, 200, 3), dtype=np.float32)
print("GPU number available:", len(tf.config.experimental.list_physical_devices("GPU")))

# These functions are implemented for Fit_Generator and data augmentation
# to store x_train and x_val into class directories
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
#
# def storeX_TrainWithLabelsForX_Test(x_testArray, testPath):
#     print("Storing image and labels")
#     for length in range(len(x_testArray)):
#         img = Image.fromarray(x_testArray[length])
#         createLabelFolders(str(testPath))
#         if os.path.isfile(str(testPath)+"/" + str(length) + ".jpg"):
#             print("File already exists, jump to next file")
#         else:
#             img.save(str(testPath)
#             +"/" + str(length) + ".jpg")
#
#     print("Finished")


# def createLabelFolders(path):
#     try:
#         os.mkdir(path)
#     except OSError:
#         print("Directory already created", path)
#     else:
#         print("Successfully created the directory %s ", path)

# Method to display image
# def displayImage(singleImage):
#     img = Image.fromarray(singleImage)
#     img.show()


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

# The following two functions are for sorting files in directory
# with int inside strings numerically
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

# Resize all images into unified size, since some photo have different size
# Resize a single image into shape 400*600*3, meaning 400px * 600px * 3 channels (RGB)
def resizeImagesAndSave(imgArray, arrayToSave, imageRangeFrom, imageRangeTo):
    # for singleImage in range(len(images)):
    print("Processing array to store images")

    scale_percent = 100  # percent of original size

    for singleImage in range(imageRangeFrom, imageRangeTo):
        print("Currently processing the", singleImage, "image")

        # All images should be cropped to this size
        cropImage = resize(image.imread(imgArray[singleImage]), (200, 200, 3), preserve_range=True)

        width = int(cropImage.shape[1] * scale_percent / 100)
        height = int(cropImage.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(cropImage, dim, interpolation=cv2.INTER_AREA)
        # Saved to specific array parameter
        resized /= 255
        arrayToSave[singleImage] = resized
        # print("Image shape after crop:", np.array(cropImage).size)
    print("Process finished for array to store images")


# Function to convert 3 channel rgb image to greyscale 1 channel Referenced from:
# https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python by Mark Amery
# def rgb2gray(rgb):
#     print("Grey-Scaling images for array")
#     r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
#     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#     print("Grey-Scaling process finished for array")
#     return gray


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
x_train_image_range_to = X_TRAIN_IMG_ARRAY_LENGTH

print("Processing y_train to constraint range")
y_train = y_train[x_train_image_range_from:x_train_image_range_to]

print("Processing array x_train to store images")
resizeImagesAndSave(x_train_images, x_train, x_train_image_range_from, x_train_image_range_to)


# The following functions are for data augmentation and fit_generator
# storeX_TrainWithLabelsForX_Test(x_test,"E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/test1/")
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
# print("Done")

# Convert x_train and x_test into greyscale img with shape (numIMG,400,600)
# print("Converting x_train into grayscale")
# x_train = rgb2gray(np.array(x_train))
# print("Converting x_test into grayscale")
# x_test = rgb2gray(np.array(x_test))

# Turn y_train to be categorical
# y_train will be 1 value initially, either class 0,1,2 ... 22 with shape (10270,)
# categorize y_train into (10270,23), which shows the most possible class
print("y_train being processed to be categorical")
y_train = to_categorical(y_train, 23)
print("Categorical process finished")


print("Finished processing y_train to constraint range")

# Keras Model Part
print("Building model...")

VGG = vgg16.VGG16(input_shape=(200, 200, 3), include_top=False, weights="imagenet",classes=23)
VGG.trainable = False
model = keras.Sequential([
    VGG,
    keras.layers.Flatten(),
    keras.layers.Dense(units=512, activation="relu"),
    keras.layers.Dense(units=256, activation="relu"),
    keras.layers.Dense(units=23, activation="softmax")
])

print("Predicting model")

model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])
model.fit(x_train, y_train, validation_split=0.3, epochs=6, batch_size=32,  shuffle=True)

model.save("my_model_butterfly_VGG16_Version2")