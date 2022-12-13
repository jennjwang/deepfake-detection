import json
import os
from distutils.dir_util import copy_tree
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import backend as K
print('TensorFlow version: ', tf.__version__)
os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/usr/lib/cuda"


dataset_path = './/complete_data//split_dataset_marked//'
tmp_debug_path = './/complete_data//tmp_debug_marked//'
print('Creating Directory: ' + tmp_debug_path)
os.makedirs(tmp_debug_path, exist_ok=True)

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

input_size = 128
batch_size_num = 32
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

train_datagen = ImageDataGenerator(
    rescale = 1/255,    #rescale the tensor values to [0,1]
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.2,
    zoom_range = 0.1,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

train_generator = train_datagen.flow_from_directory(
    directory = train_path,
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = "binary",  #"categorical", "binary", "sparse", "input"
    batch_size = batch_size_num,
    shuffle = True
)

val_datagen = ImageDataGenerator(
    rescale = 1/255    #rescale the tensor values to [0,1]
)

val_generator = val_datagen.flow_from_directory(
    directory = val_path,
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = "binary",  #"categorical", "binary", "sparse", "input"
    batch_size = batch_size_num,
    shuffle = True
)

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

# Train a CNN classifier
efficient_net = EfficientNetB0(
    weights = 'imagenet',
    input_shape = (input_size, input_size, 3),
    include_top = False,
    pooling = 'max'
)


model = Sequential()
model.add(efficient_net)
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.summary()

# Compile model
model.compile(optimizer = Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_filepath = './/complete_data//tmp_checkpoint_marked'
print('Creating Directory: ' + checkpoint_filepath)
os.makedirs(checkpoint_filepath, exist_ok=True)

custom_callbacks = [
    EarlyStopping(
        monitor = 'val_accuracy',
        mode = 'max',
        patience = 7,
        verbose = 1
    ),
    ModelCheckpoint(
        filepath = os.path.join(checkpoint_filepath, 'best_model.h5'),
        monitor = 'val_accuracy',
        mode = 'max',
        verbose = 1,
        save_best_only = True
    )
]

# Train network
num_epochs = 20
history = model.fit(
    train_generator,
    epochs = num_epochs,
    steps_per_epoch = len(train_generator),
    validation_data = val_generator,
    validation_steps = len(val_generator),
    callbacks = custom_callbacks
)
print("printing history: \n")
print(history.history)

# Plot results
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig("/home/ewang96/CS1430/Editing_CVFinal/CVFinal/complete_data/Accuracy_marked.png")
plt.clf()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("/home/ewang96/CS1430/Editing_CVFinal/CVFinal/complete_data/Loss_marked.png")
plt.clf()


# load the saved model that is considered the best
best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.h5'))

# Generate predictions
test_generator.reset()

preds = best_model.predict(
    test_generator,
    verbose = 1
)

test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten()
})
print(test_results.to_string())

test_results.to_json("./complete_data/cnn_preds_values_marked.json", orient="values")


test_generator_class = test_datagen.flow_from_directory(
    directory = test_path,
    classes=['real', 'fake'],
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = "binary",
    batch_size = 1,
    shuffle = False
)

labels_flipped = test_generator_class.classes
predictions = preds.flatten()
correct = 0
total = 0
correct_ones = []
rounded_preds = []
correct_labels = []
for i in range(len(labels_flipped)):
    rounded = 0
    if predictions[i] >= 0.5:
        rounded = 1
    rounded_preds.append(rounded)
    if rounded != labels_flipped[i]:
        correct += 1
        correct_ones.append(1)
    else:
        correct_ones.append(0)
    
    if labels_flipped[i] == 1:
        correct_labels.append(0)
    else:
        correct_labels.append(1)
    
    total += 1
print("Total Correct: ", correct)
print("Total: ", total)
accuracy = correct / total
print("Accuracy: ", accuracy)
print(correct_ones)

print(test_results.shape)
print(test_results.to_string())

confusion_matrix = metrics.confusion_matrix(correct_labels, rounded_preds)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Fake", "Real"])
cm_display.plot()
plt.title("Confusion Matrix for Predictions on Faces")
plt.savefig("/home/ewang96/CS1430/Editing_CVFinal/CVFinal/complete_data/Confusion_marked.png")
plt.clf()
