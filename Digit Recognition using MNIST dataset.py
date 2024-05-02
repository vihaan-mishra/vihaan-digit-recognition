#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import numpy as np    # library using matrices
import matplotlib.pyplot as plt    # library for data visualization (in this case loading the digit)
import tensorflow as tf    # open-source library used for machine learning

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()    # load data, 50/50 for training and testing

x_train = tf.keras.utils.normalize(x_train, axis=1)    # rescale attributes from 0 to 1 
x_test = tf.keras.utils.normalize(x_test, axis=1)    

# keras is an API used mostly used for neural networks
# and API is a way for multiple computer models and programs to communicate with ecah other

#model = tf.keras.models.Sequential()    # a linear stack of layers to get each pixel, added below
#model.add(tf.keras.layers.Flatten(input_shape=(28,28)))    # flatten 28x28 pixel digit into individual neurons
#model.add(tf.keras.layers.Dense(128, activation='relu'))    # add layer to minimize
#model.add(tf.keras.layers.Dense(128, activation='relu'))    # add layer to minimize
#model.add(tf.keras.layers.Dense(10, activation='softmax'))    # minimize to one of the 10 digits

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model.fitt(x_train, y_train, epochs=3)

#model.save('handwritten.model')    # Save trained model

model = tf.keras.models.load_model('handwritten.model')    # Load the trained model

loss, accuracy = model.evaluate(x_test, y_test)    # Test the model and calculates accuracy and losses (num incorrect)

print(loss)
print(accuracy)

### Next steps

# CNN - Convelutional Neural Networks
# Data Augmentation

# After top 2 change dataset


# In[2]:


pip install opencv-python numpy matplotlib tensorflow

