import numpy
import pandas
from tensorflow.python.keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

model = load_model("../my_model_butterfly_VGG16_fitGen_fine-tuned_2")

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    # rescale=1. / 255
    )

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    # rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    'E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/data_split_new/train',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    directory=r"E:/360MoveData/Users/11047/Desktop/Aritificial Intelligence/Coursework/test1/",
    target_size=(200, 200),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False
)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=numpy.argmax(pred,axis=1)
labels = train_generator.class_indices
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pandas.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("PredictGeneratorPredictionResults.csv",index=False)