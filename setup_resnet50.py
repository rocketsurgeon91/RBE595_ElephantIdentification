"""
Created on Sun Apr 18 12:24:05 2021

@author: fjbar
"""
import keras

model = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
#Code from: https://faroit.com/keras-docs/2.0.6/applications/#resnet50
model.save("my_model.h5")
