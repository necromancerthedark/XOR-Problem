import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pygame

pygame.init()


def clearScreen():
    try:
        os.system("cls")
    except:
        os.system("clear")


y_axis = []
width = 400
height = 400
disp = []
win = pygame.display.set_mode((width, height))

for i in range(height):
    for j in range(width):
        y_axis.append([i/height, j/width])

new_model = keras.models.load_model('Xor_Problem.h5')
clearScreen()
y_axis = tf.constant(y_axis)
predicted = new_model.predict(y_axis)


for i in range(400):
    temp = []
    for j in range(i*400, i*400+400):
        a = predicted[j]*255
        temp.append(a)
    disp.append(temp)

pixels = np.array([np.array(xi) for xi in disp])
pixels2 = pixels.reshape(400, -1)
print(pixels2.shape)

pygame.surfarray.blit_array(win, pixels2)

pygame.display.update()

input()

pygame.quit()
