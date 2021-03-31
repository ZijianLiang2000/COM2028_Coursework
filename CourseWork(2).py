import re
import time

import matplotlib.pyplot as plt
import tensorflow.keras as keras
from keras.models import load_model
from tensorflow.keras import applications as applications
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input



startTime = time.perf_counter()
print("Start execution time:", startTime)

X_TRAIN_IMG_ARRAY_LENGTH = 10270

X_TEST_IMG_ARRAY_LENGTH = 10

# Initialize arrays
readLine = []
# create y_train to store y_train data
y_train = []
# store image pixel data for x_train and x_test
# x_train = []
# x_test = []

print("GPU number available:", len(tf.config.experimental.list_physical_devices("GPU")))

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1. / 255)

IMAGE_SIZE = (200, 200)
BATCH_SIZE = 64

train_generator = train_datagen.flow_from_directory(
    'E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/data_split/train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    'E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/data_split/validation',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Image generator format: float32, Min:0, Max:1

VGG = applications.VGG16(input_shape=(200, 200, 3), include_top=False, weights="imagenet",classes=23)
VGG.trainable = False
model = keras.Sequential([
    VGG,
    keras.layers.Flatten(),
    keras.layers.Dense(units=512, activation="relu"),
    keras.layers.Dense(units=256, activation="relu"),
    keras.layers.Dense(units=23, activation="softmax")
])

model.summary()



# model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])

hist = model.fit_generator(generator=train_generator, validation_data=validation_generator,
                           steps_per_epoch=8216 // BATCH_SIZE,
                           epochs=50,
                           validation_steps=2054 // BATCH_SIZE)

model.save("OptimalModel2")

# print("Predicting model")
# model1 = load_model("OptimalModel")
# y_pred = model.predict(x_test)
# y_pred = np.argmax(y_pred, axis=1)
# print(y_pred)
# np.savetxt("saved_numpy_data2.csv", y_pred.flatten().tolist(), delimiter=",")