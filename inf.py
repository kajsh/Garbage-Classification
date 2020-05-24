# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 02:56:46 2020

@author: Vishwa
"""
import cv2
import tensorflow as tf
import numpy as np

filepath1 = "./dataset/plastic/plastic501.jpg"

def prepare(filepath):
    arr = cv2.imread(filepath)
    arr2 = cv2.resize(arr, (50, 50))
    return arr2.reshape(-1, 50, 50, 3)

model = tf.keras.models.load_model('saved_model')
#model.summary()

prediction = model.predict(prepare(filepath1))
print(prediction)
a = np.argmax(prediction[0])
#print(a)

if a==0:
    print("Cardboard")
elif a==1:
    print("Paper")
elif a==2:
    print("Plastic")
elif a==3:
    print("Metal")
else:
    print("Unconclusive")