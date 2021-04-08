import cv2
import pandas
from PIL.Image import fromarray
from keras.models import load_model
import glob
import re
import numpy as np
import tensorflow as tf
from matplotlib import image
# Print time for start execution of code
from skimage.transform import resize
from keras.applications.vgg16 import preprocess_input

X_TEST_IMG_ARRAY_LENGTH_Part1 = 8000
X_TEST_IMG_ARRAY_LENGTH_Part2 = 15009

# Initialize arrays
readLine = []

x_test_part1 = np.empty((X_TEST_IMG_ARRAY_LENGTH_Part1, 200, 200, 3), dtype=np.uint8)
x_test_part2 = np.empty((X_TEST_IMG_ARRAY_LENGTH_Part2 - X_TEST_IMG_ARRAY_LENGTH_Part1, 200, 200, 3), dtype=np.uint8)

print("GPU number available:", len(tf.config.experimental.list_physical_devices("GPU")))

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
        resized = preprocess_input(resized)
        arrayToSave[singleImage] = resized
        # print("Image shape after crop:", np.array(cropImage).size)
    print("Process finished for array to store images")

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

# Load x_test images in local directory
x_test_images = glob.glob("E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/test/test" + '/*.jpg')
x_test_images.sort(key=natural_keys)

# Split x_test image range into two parts to predict
# so it does not occupy too much RAM leading to the system crashing
x_test_image_part1_range_from = 0
x_test_image_part1_range_to = X_TEST_IMG_ARRAY_LENGTH_Part1
x_test_image_part2_range_from = X_TEST_IMG_ARRAY_LENGTH_Part1
x_test_image_part2_range_to = X_TEST_IMG_ARRAY_LENGTH_Part2

x_test_images_part1 = x_test_images[x_test_image_part1_range_from:x_test_image_part1_range_to]
x_test_images_part2 = x_test_images[x_test_image_part2_range_from:x_test_image_part2_range_to]

print("Processing array x_test part1 to store images")
resizeImagesAndSave(x_test_images_part1, x_test_part1, 0, len(x_test_images_part1))
print("Processing array x_test part2 to store images")
resizeImagesAndSave(x_test_images_part2, x_test_part2, 0, len(x_test_images_part2))
# print("Pre-processing x_test_part1")
# x_test_part1 = preprocess_input(x_test_images_part1)
# print("Pre-processing x_test_part2")
# x_test_part2 = preprocess_input(x_test_images_part2)
# Can change models

# Original Model to use
# model = load_model("my_model_butterfly_VGG16_fitGen_fine-tuned_2")
model = load_model("../my_model_butterfly_VGG16_fitGen_fine-tuned_2")
print("Predicting x_test_part1")
y_pred_part1 = model.predict(x_test_part1)
y_pred_part1 = np.argmax(y_pred_part1,axis=1)

print("Predicting x_test_part2")
y_pred_part2 = model.predict(x_test_part2)
y_pred_part2 = np.argmax(y_pred_part2,axis=1)

print("Creating csv files")
lengthArray_part1 = []
for i in range(15009):
    lengthArray_part1.append(i)

df1 = pandas.DataFrame({"id": lengthArray_part1})
df1.to_csv("y_pred_VGG16fineTuned.csv", index=False)

df2 = pandas.DataFrame({"label": y_pred_part1})
df2.to_csv("y_pred_VGG16fineTuned_part1.csv", index=False)

df3 = pandas.DataFrame({"label": y_pred_part2})
df3.to_csv("y_pred_VGG16fineTuned_part2.csv", index=False)
