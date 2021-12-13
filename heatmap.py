## Visualization of the model
## Codes refer and adapted from Practical Guide for Visualizing CNNs Using Saliency Maps
## at https://towardsdatascience.com/practical-guide-for-visualizing-cnns-using-saliency-maps-4d1c2e13aeca

## refer and adapted from keras-vis document
## at https://keisen.github.io/tf-keras-vis-docs/examples/attentions.html

from random import randint
import matplotlib.pylab as plt 
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras 
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras 
from keras.models import Sequential, Model

workpath = 'D:/543project/'
bird_classes = pd.read_csv(workpath+'class_dict.csv')
bird_species = pd.read_csv(workpath+'Bird Species.csv')
bird_species.sample(5)
birdTypes = os.listdir(workpath+'train')
species = np.array(bird_classes['class'])

fig_path = "D:/543project/train/ALBATROSS/002.jpg"
test_img = plt.imread(fig_path)
test_img = np.array(test_img).astype('float32')
test_img /= 255.0
test_img1 = test_img

train_data = ImageDataGenerator(preprocessing_function = preprocess_input, rotation_range=20, horizontal_flip=True)
train_generator = train_data.flow_from_directory(workpath+'train',
                                                batch_size=64,
                                                target_size=(224,224),
                                                class_mode='categorical')
valid_data = ImageDataGenerator(preprocessing_function = preprocess_input, rotation_range=20, horizontal_flip=True)
validation_generator = valid_data.flow_from_directory(workpath+'valid',
                                                batch_size=64,
                                                target_size=(224,224),
                                                class_mode='categorical')

x_test = []
x_test.append(test_img)

workpath = 'D:/543project/'
vgg_model = keras.models.load_model('./my_model3')
print("loaded")

  
def get_feature_maps(model, layer_id, input_image):
    model_ = Model(inputs=[model.input], outputs=[model.layers[layer_id].output])
    return model_.predict(np.expand_dims(input_image, 
                                         axis=0))[0,:,:,:]
  
def plot_features_map(img_idx=None, layer_idx=[0, 2, 4, 6, 8, 10, 12, 16], 
                      x_test=x_test, cnn=vgg_model):
    img_idx = 0
    input_image = x_test[img_idx]
    fig, ax = plt.subplots(3,3,figsize=(10,10))
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    ax[0][0].imshow(input_image)
    ax[0][0].set_title('original img ')
    for i, l in enumerate(layer_idx):
        feature_map = get_feature_maps(cnn, l, input_image)
        ax[(i+1)//3][(i+1)%3].imshow(feature_map[:,:,0])
        ax[(i+1)//3][(i+1)%3].set_title('layer {} - {}'.format(l, 
                                                               cnn.layers[l].get_config()['name']))
    return img_idx

def plot_features_map_layers(img_idx=None, layer_idx=0, 
                      x_test=x_test, cnn=vgg_model):
    img_idx = 0
    input_image = x_test[img_idx]
    plt_num = 4
    fig_size = int(10/3*plt_num)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=2.0, wspace=None, hspace=None)
    fig, ax = plt.subplots(plt_num,plt_num,figsize=(fig_size,fig_size))
    ax[0][0].imshow(input_image)
##    ax[0][0].set_title('original img, with layer:'+str(layer_idx))
    for i in range(15):
        feature_map = get_feature_maps(cnn, layer_idx, input_image)
        num = randint(0,feature_map.shape[2]-1)
        ax[(i+1)//plt_num][(i+1)%plt_num].imshow(feature_map[:,:,num])
##        ax[(i+1)//plt_num][(i+1)%plt_num].set_title('at channel:'+str(num))
    return img_idx

vgg_model.summary()
layer_idx=[0, 2, 4, 6, 8, 10, 12, 16]
##plot_features_map()
plot_features_map_layers(layer_idx = 7)
plt.show()




