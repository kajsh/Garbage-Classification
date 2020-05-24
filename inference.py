# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:17:55 2020

@author: Kajol
"""

import cv2
import tensorflow as tf

filepath1 = "./dataset/cardboard/cardboard3.jpg"

def prepare(filepath):
    arr = cv2.imread(filepath)
    return arr.reshape(-1, 50, 50, 3)

model = tf.keras.models.load_model('my_model5050.h5')
model.summary()

prediction = model.predict([prepare(filepath1)])
print(max(prediction))
