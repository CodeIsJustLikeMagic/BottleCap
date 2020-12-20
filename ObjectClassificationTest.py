import tensorflow as tf
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from numpy import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg

IMG_WIDTH=200
IMG_HEIGHT=200
def create_dataset(img_folder):
    img_data_array = []
    class_name = []

    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1, file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name

img_data, class_name = create_dataset(r'TensorImages')
target_dict={k: v for v, k in enumerate(np.unique(class_name))}
print(target_dict)
target_val=  [target_dict[class_name[i]] for i in range(len(class_name))]
print(len(target_val))
model=tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(IMG_HEIGHT,IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4)
        ])
print(model.summary())
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=tf.cast(np.array(img_data), tf.float64), y=tf.cast(list(map(int,target_val)),tf.int32), epochs=20)
print(history)

#test prediction with one image
image_path = r'TensorImages/FaceUps/faceUp (1).png'
def testimage(image_path):
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
    image = np.array(image)
    image = image.astype('float32')
    image /= 255
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    print(predictions)
    m = np.argmax(predictions[0])
    print(image_path, 'is predicted to be class:', np.unique(class_name)[m])
testimage(image_path)
testimage(r'TensorImages/FaceUps/faceUp (2).png')