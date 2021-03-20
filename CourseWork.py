import cv2
import keras
import numpy as np
from PIL import Image
from numpy.random import seed
from numpy import asarray
from matplotlib import image
from matplotlib import pyplot
import tensorflow as tf
import pandas
import glob
import re
import os

# Initialize arrays
readLine = []
# create y_train to store y_train data
y_train = []
# store image pixel data for x_train and x_test
x_train = []
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

# Example to open a photo according to directory
# image = Image.open("E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/train/train/0.jpg")
# imageData = asarray(image)
# print(imageData.shape)

# Refernce from https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# Load x_train images in local directory
x_train_images = glob.glob("E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/train/train" + '/*.jpg')
x_train_images.sort(key=natural_keys)

# Load x_test images in local directory
x_test_images = glob.glob("E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/test/test" + '/*.jpg')
x_test_images.sort(key=natural_keys)

# Resize all images into unified size, since some photo have different size
# Resize a single image into shape 400*600*3, meaning 400px * 600px * 3 channels (RGB)
def resizeImagesAndSave(imgArray,arrayToSave,imageRange):
    # for singleImage in range(len(images)):
    for singleImage in range(imageRange):
        # print("Current image shape:",np.array(image.imread(imgArray[singleImage])).shape)

        # All images should be cropped to this size
        cropImage = np.resize(image.imread(imgArray[singleImage]),(400,600,3))
        # Saved to specific array parameter
        arrayToSave.append(cropImage)
        # print("Image shape after crop:", np.array(cropImage).size)

# Function to convert 3 channel rgb image to greyscale 1 channel
# Referenced from: https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python by Mark Amery
def rgb2gray(rgb):

    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def reshapeArrayImage(array):
    np.array(array).reshape(-1,400,600,1)

# Run to open and load y_train in local terminal
openY_Train("train.txt")
loadY_Train()

# Resize x_train and x_test images into unified size and save into array, with range of images constrained
resizeImagesAndSave(x_train_images,x_train,5)
resizeImagesAndSave(x_test_images,x_test,3)

# They needed to be normalized within 255
x_train = np.array(x_train)/255
x_test = np.array(x_test)/255

# Convert x_train and x_test into greyscale img with shape (numIMG,400,600)
x_train = rgb2gray(np.array(x_train))
x_test = rgb2gray(np.array(x_test))

# Reshape x_train and x_test images into -1,400,600,1
reshapeArrayImage(x_train)
reshapeArrayImage(x_test)

# Now shape of x_train is (500, 400, 600)
# shape of x_test is(300, 400, 600)



