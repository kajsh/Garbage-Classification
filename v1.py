# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 20:20:16 2020

@author: Kajol
"""

import numpy as np
import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential

#Load data

image_count = 2491
main_dir = os.path.join(os.path.dirname('.\dataset'), 'dataset')
train_dir = os.path.join(main_dir, 'train')
test_dir = os.path.join(main_dir, 'test')

BATCH_SIZE = 24
IMG_HEIGHT = 384
IMG_WIDTH = 512
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
#print(STEPS_PER_EPOCH)
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data_gen = train_generator.flow_from_directory(directory=train_dir,
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_data_gen = test_generator.flow_from_directory(directory=test_dir,
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))

#Sample Training Images
'''
sample_training_images, _ = next(train_data_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
plotImages(sample_training_images[:5])
'''

#Creating Model

model = Sequential([
    Conv2D(16, 3, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(5)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit(
    train_data_gen,
    epochs=6,
    validation_data=test_data_gen,
)
                        
#model.save('my_model.h5') 
