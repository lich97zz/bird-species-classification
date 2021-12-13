#use vgg16 pretained model to fit bird species detection task
#codes adapted from:
#https://www.kaggle.com/blakewinters/vgg16-bird-species
#original author: Blake Summers

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10*1024)])
  except RuntimeError as e:
    print(e)

def train():

  with tf.device('/gpu:0'):
      workpath = 'D:/543project/'
      bird_classes = pd.read_csv(workpath+'class_dict.csv')
      bird_species = pd.read_csv(workpath+'Bird Species.csv')
      bird_species.sample(5)
      birdTypes = os.listdir(workpath+'train')
      species = np.array(bird_classes['class'])

      def show_sample(df, num, species):
          import matplotlib.pyplot as plt
          df = df.sample(12)
          plt.figure(figsize=(20,20))
          for i in range(len(df)):
              path = workpath+ df.iloc[i]['filepaths']
              label = df.iloc[i]['labels']
              Type = df.iloc[i]['data set']
              plt.subplot(3, 4, i+1)
              img = tf.keras.preprocessing.image.load_img(path)
              plt.imshow(img)
              plt.xlabel(f'{label} : {Type}')
          plt.show()

##      show_sample(bird_species, 5, bird_species)
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
      def show_preprocessing_result(train_generator):
          import matplotlib.pyplot as plt
          for _ in range(3):
              img, label = train_generator.next()
              print(img.shape) 
              plt.imshow(img[0])
              plt.show()

      show_preprocessing_result(train_generator)
      vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
      vgg_model.trainable = False

      layer0 = tf.keras.layers.Flatten(name='flatten')(vgg_model.output)
      layer1 = tf.keras.layers.Dense(4096, activation='relu',name='fc1')(layer0)
      layer2 = tf.keras.layers.Dense(4096, activation='relu',name='fc2')(layer1)
      out_layer = tf.keras.layers.Dense(315, activation='softmax')(layer2)
      vgg_model = tf.keras.Model(vgg_model.input, out_layer)

##      vgg_model.summary()
      
      print("training 1")
      opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
      vgg_model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
      callbacks = [EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]

      history = vgg_model.fit(
            train_generator, 
            epochs=10,
            verbose=1,
            validation_data = validation_generator,
            callbacks=callbacks)


      vgg_model.save('my_model1')

      print("training 2")
      opt = tf.keras.optimizers.Adam(learning_rate=0.00005)
      vgg_model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
      callbacks = [EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]

      history = vgg_model.fit(
            train_generator, 
            epochs=15,
            verbose=1,
            validation_data = validation_generator,
            callbacks=callbacks)

##      vgg_model.save('my_model2')

      import matplotlib.pyplot as plt
      acc = history.history['accuracy']
      val_acc = history.history['val_accuracy']
      loss = history.history['loss']
      val_loss = history.history['val_loss']

      epochs = range(len(acc))

      plt.plot(epochs, acc, 'r', label='Training accuracy')
      plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
      plt.title('Training and validation accuracy')
      plt.legend(loc=0)
      plt.figure()

      plt.show()






workpath = 'D:/543project/'
train()
vgg_model = keras.models.load_model('./my_model2')
print("loaded")
test = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test.flow_from_directory(
        workpath+'test',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical')

print("test generated")
vgg_model.evaluate(test_generator,use_multiprocessing=False)


