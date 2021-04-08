import datetime

from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Model, load_model
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import SGD
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping

print("Running fine tuning")

# model = load_model("my_model_butterfly_VGG16_FitGenerator_Latest")
model = load_model("../Previous_Files/my_model_butterfly_VGG16_fitGen_fine-tuned")

# Step 1 - Set up fine tuning on pre-trained ImageNet vgg19 model - train all layers for VGG16 and VGG19 models but only the Layers from
# 94 and above for the Inception V3 and Xception models
for layer in model.layers:
    layer.trainable = True

# Step 2 - Compile the revised model using SGD optimizer with a learing rate of 0.0001 and a momentum of 0.9
model.compile(optimizer=SGD(lr=0.00001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

BATCH_SIZE = 16
IMAGE_SIZE = (200, 200)


print("Fine-Tuning current model")

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

# Step 3 - Fit the revised model, log the results and the training time
now = datetime.datetime.now
t = now()

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

fine_tuning_history = model.fit_generator(
    train_generator,
    epochs = 15,
    steps_per_epoch=8216 // BATCH_SIZE,
    validation_data=validation_generator,
    # number of val samples
    validation_steps=2054 // BATCH_SIZE,
    callbacks=[checkpoint,early])

print('Training time: %s' % (now() - t))

model_name = "my_model_butterfly_VGG16_fitGen_fine-tuned_2"
print("Saving model")
model.save(str(model_name))
print("Model saved as", str(model_name))


# Plot accuracy and loss
plt.plot(fine_tuning_history.history["accuracy"])
plt.plot(fine_tuning_history.history["val_accuracy"])
plt.title("model accuracy")
plt.xlabel("epoch")
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig("AccuracyAndVal_Accuracy_butterfly_VGG16_fitGen_fine-tuned.png")

plt.plot(fine_tuning_history.history['loss'])
plt.plot(fine_tuning_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig("LossAndVal_Loss_butterfly_VGG16_fitGen_fine-tuned.png")
