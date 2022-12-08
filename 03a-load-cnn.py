import json
import os
from distutils.dir_util import copy_tree
import shutil
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from efficientnet.tfkeras import EfficientNetB0 #EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

print("loading previous best model to test this current set!")
input_size = 128
dataset_path = './/data01_split_dataset//'
test_path = os.path.join(dataset_path, 'test')
checkpoint_filepath = './/tmp_checkpoint//'

test_datagen = ImageDataGenerator(
    rescale = 1/255    #rescale the tensor values to [0,1]
)

test_generator = test_datagen.flow_from_directory(
    directory = test_path,
    classes=['real', 'fake'],
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = None,
    batch_size = 1,
    shuffle = False
)

test_generator_class = test_datagen.flow_from_directory(
    directory = test_path,
    classes=['real', 'fake'],
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = "binary",
    batch_size = 1,
    shuffle = False
)

best_model = load_model(os.path.join(checkpoint_filepath, 'saved_model.pb'))
# best_model = load_model(os.path.join(checkpoint_filepath))

# Generate predictions
test_generator.reset()

preds = best_model.predict(
    test_generator,
    verbose = 1
)
X_test, y_test = next(test_generator_class)


labels_flipped = test_generator_class.classes

test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten()
})

predictions = preds.flatten()
correct = 0
total = 0
correct_ones = []
for i in range(len(labels_flipped)):
    rounded = 0
    if predictions[i] >= 0.5:
        rounded = 1
    if rounded != labels_flipped[i]:
        correct += 1
        correct_ones.append(1)
    else:
        correct_ones.append(0)
    total += 1
print("Total Correct: ", correct)
print("Total: ", total)
accuracy = correct / total
print("Accuracy: ", accuracy)
print(correct_ones)

print(test_results.shape)
print(test_results.to_string())

# result = test_results.to_json(orient="records").replace('},{', '} {')

# with open('./cnn_preds.json', 'w') as outfile:
#     json.dump(result, outfile, indent=4)

test_results.to_json("./01_loaded_model_test.json", orient="values")
