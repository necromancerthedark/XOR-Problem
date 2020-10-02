# imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pygame
import os
import numpy as np


# function for clearing terminal screen
def clearScreen():
    try:
        os.system("cls")
    except:
        os.system("clear")


# variables
y_axis = []
width = 400
height = 400
winloop = True
disp = []
predicted = []
black = (0, 0, 0)

# pygame window initialization
pygame.init()
win = pygame.display.set_mode((width, height))

# normalization of coordinates of pygame window between 0 and 1
for i in range(height):
    for j in range(width):
        y_axis.append([i/height, j/width])

# creating sequential model
input_layer = layers.Dense(2, activation='relu', name='input_layer')
hidden_layer = layers.Dense(4, activation='relu', name='hidden_layer')
output_layer = layers.Dense(1, activation='sigmoid', name='output_layer')
model = keras.Sequential()
model.add(keras.Input(shape=(2, )))
model.add(input_layer)
model.add(hidden_layer)
model.add(output_layer)

# input data
x_train = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]])

# output data
y_train = tf.constant([[0], [1], [1], [0]])
clearScreen()

# converting normalized coordinates to tensor
y_axis = tf.constant(y_axis)

# compiling the model
model.compile(optimizer="adam", loss="mse")

# animation begin
while winloop:

    disp = []
    win.fill(black)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            winloop = False
    for _ in range(10):
        model.fit(x_train, y_train, epochs=10)  # training the model

    # predicting result from normalized coordinates
    predicted = model.predict(y_axis)

    # converting predicted values to pygame undestandable format
    for i in range(height):
        temp = []
        for j in range(i*width, i*width+width):
            a = predicted[j]*200
            temp.append(a)
        disp.append(temp)
    pixels = np.array([np.array(xi) for xi in disp])
    pixels2 = pixels.reshape(400, -1)

    pygame.surfarray.blit_array(win, pixels2)  # drawing array on pygame window

    pygame.display.update()
