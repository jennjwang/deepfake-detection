import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

from skimage.transform import resize

from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

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

def preprocess_fn(img):
    """ Preprocess function for ImageDataGenerator. """

    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def LIME_explainer(model, path, preprocess_fn, count):
    def image_and_mask(title, count, positive_only=True, num_features=5,
                       hide_rest=True):
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=positive_only,
            num_features=num_features, hide_rest=hide_rest)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.title(title)
        plt.savefig("/home/ewang96/CS1430/Editing_CVFinal/CVFinal/complete_data/imagemask" + str(count) + ".jpg")
        plt.clf()

    # Read the image and preprocess it as before
    image = imread(path)
    image = resize(image, (128, 128, 3), preserve_range=True)
    image = preprocess_fn(image)
    

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image.astype('double'), model.predict)

    # The top 5 superpixels that are most positive towards the class with the
    # rest of the image hidden
    image_and_mask("Top 5 superpixels", count, positive_only=True, num_features=5,
                   hide_rest=True)
    count += 1
    # The top 5 superpixels with the rest of the image present
    image_and_mask("Top 5 with the rest of the image present",count, 
                   positive_only=True, num_features=5, hide_rest=False)
    count += 1
    # The 'pros and cons' (pros in green, cons in red)
    image_and_mask("Pros(green) and Cons(red)", count, 
                   positive_only=False, num_features=10, hide_rest=False)
    count += 1
    # Select the same class explained on the figures above.
    ind = explanation.top_labels[0]
    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    plt.title("Map each explanation weight to the corresponding superpixel")
    plt.savefig("/home/ewang96/CS1430/Editing_CVFinal/CVFinal/complete_data/superpixels.jpg")
    plt.clf()

checkpoint_filepath = "/home/ewang96/CS1430/Editing_CVFinal/CVFinal/complete_data/2534309"

best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.h5'))

test_datagen = ImageDataGenerator(
    rescale = 1/255    #rescale the tensor values to [0,1]
)

dataset_path = './/complete_data//split_dataset//'
test_path = os.path.join(dataset_path, 'test')

test_generator = test_datagen.flow_from_directory(
    directory = test_path,
    classes=['real', 'fake'],
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = None,
    batch_size = 1,
    shuffle = False
)
best_model.evaluate(
    x=test_generator,
    verbose=1,
)

count = 1
# 1: true fake
# path = "/home/ewang96/CS1430/Editing_CVFinal/CVFinal/complete_data/split_dataset/test/fake/aimzesksew-005-00.png" 
# 2: false true 
path = "/home/ewang96/CS1430/Editing_CVFinal/CVFinal/complete_data/split_dataset/test/fake/aimzesksew-008-00.png" 
# # 3: false fake
# path = "/home/ewang96/CS1430/Editing_CVFinal/CVFinal/complete_data/split_dataset/test/real/dxfdovivlw-003-00.png" 
# # 4: true real
# path = "/home/ewang96/CS1430/Editing_CVFinal/CVFinal/complete_data/split_dataset/test/real/dxfdovivlw-006-00.png" 



LIME_explainer(best_model, path, preprocess_fn, count)