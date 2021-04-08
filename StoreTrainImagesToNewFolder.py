import os

import cv2
import glob
import re
import time

from PIL.Image import fromarray
from keras.applications import vgg16 as vgg16
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers as layers
from matplotlib import image
# Print time for start execution of code
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler

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
# standardScaler = StandardScaler()

x_train = np.empty((X_TRAIN_IMG_ARRAY_LENGTH, 200, 200, 3), dtype=np.uint8)
print("GPU number available:", len(tf.config.experimental.list_physical_devices("GPU")))


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

print("Executing Coursework_train file")


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
        arrayToSave[singleImage] = resized
        # print("Image shape after crop:", np.array(cropImage).size)
    print("Process finished for array to store images")



# Run to open and load y_train in local terminal
openY_Train("../Previous_Files/train.txt")
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


print(np.array(x_train[0]))
img = fromarray(x_train[0])
img.show()

X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=13)



# These functions are implemented for Fit_Generator and data augmentation
# to store x_train and x_val into class directories
def storeX_TrainWithLabels(x_trainArray, y_trainArray, x_validArray, y_validArray, trainPath, validPath):
    print("Storing image and labels")
    for length in range(len(x_trainArray)):
        for y_range in range(23):
            if int(y_trainArray[length]) == y_range:
                img = fromarray(x_trainArray[length])
                createLabelFolders(str(trainPath) + str(y_range))
                if os.path.isfile(str(trainPath) + str(y_range) + "/" + str(length) + ".jpg"):
                    print("File already exists, jump to next file")
                else:
                    img.save(str(trainPath) + str(
                        y_range) + "/" + str(length) + ".jpg")
    for length in range(len(x_validArray)):
        for y_range in range(23):
            if int(y_validArray[length]) == y_range:
                img = fromarray(x_validArray[length])
                createLabelFolders(str(validPath) + str(y_range))
                if os.path.isfile(str(validPath) + str(y_range) + "/" + str(length) + ".jpg"):
                    print("File already exists, jump to next file")
                else:
                    img.save(str(validPath) + str(
                        y_range) + "/" + str(length) + ".jpg")
    print("Finished")


def createLabelFolders(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Directory already created", path)
    else:
        print("Successfully created the directory %s ", path)


def createLabelFolders(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Directory already created", path)
    else:
        print("Successfully created the directory %s ", path)


# The following functions are for data augmentation and fit_generator
storeX_TrainWithLabels(X_train, y_train, X_val, y_val,
                       "E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/data_split_new/train/",
                       "E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/data_split_new/validation/")

