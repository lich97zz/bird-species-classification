## Visualization of the model
## Codes refer and adapted from Practical Guide for Visualizing CNNs Using Saliency Maps
## at https://towardsdatascience.com/practical-guide-for-visualizing-cnns-using-saliency-maps-4d1c2e13aeca

## refer and adapted from keras-vis document
## at https://keisen.github.io/tf-keras-vis-docs/examples/attentions.html
import keras
import matplotlib.pylab as plt 
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import CategoricalScore
from tensorflow.keras.applications.vgg16 import preprocess_input
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tensorflow.keras.applications.vgg16 import VGG16 as Model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
import os
import tensorflow as tf
import pandas as pd
import numpy as np

fig_path = "D:/543project/train/ALBATROSS/002.jpg"
test_img = plt.imread(fig_path)
test_img = np.array(test_img).astype('float32')
test_img /= 255.0
test_img1 = test_img

workpath = 'D:/543project/'
vgg_model = keras.models.load_model('./my_model3')
print("loaded")

model = vgg_model

image_titles = ['ALBATROSS']
test_img = load_img(fig_path, target_size=(224, 224))
test_img = plt.imread(fig_path)
test_img = img_to_array(test_img)
test_img = test_img.reshape((1, test_img.shape[0], test_img.shape[1], test_img.shape[2]))
test_img = preprocess_input(test_img)

### 2 is the model index corresponding to ALBATROSS.
##fig_path = "D:/543project/train/ALBATROSS/001.jpg"
##test_img = load_img(fig_path, target_size=(224, 224))
##test_img = plt.imread(fig_path)
##test_img = img_to_array(test_img)
##test_img = test_img.reshape((1, test_img.shape[0], test_img.shape[1], test_img.shape[2]))
##test_img = preprocess_input(test_img)
##yhat = vgg_model.predict(test_img)
##print(yhat.argmax)
def show_saliency():
    replace2linear = ReplaceToLinear()
    score = CategoricalScore([2])
    X = preprocess_input(test_img)

    saliency = Saliency(model,
                        model_modifier=replace2linear,
                        clone=True)
    saliency_map = saliency(score, X)

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax[0].imshow(test_img1, cmap='jet')
    ax[0].axis('off')
    ax[1].imshow(saliency_map[0], cmap='jet')
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()

show_saliency()
