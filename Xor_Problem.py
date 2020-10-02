# this model only trains and saves the model after training
# you can adjust training parameter by changing value of epochs in variable
# for more comments see file XOR_Problem_Animation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

epochs = 10

input_layer = layers.Dense(2, activation='relu', name='input_layer')
hidden_layer = layers.Dense(4, activation='relu', name='hidden_layer')
output_layer = layers.Dense(1, activation='sigmoid', name='output_layer')


model = keras.Sequential()
model.add(keras.Input(shape=(2, )))

model.add(input_layer)
model.add(hidden_layer)
model.add(output_layer)

x_train = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = tf.constant([[0], [1], [1], [0]])
print(x_train.shape)
y_pred = tf.constant([[0, 1], [0.9, 0], [0.5, 0.5], [0.1, 0.3]])
model.compile(optimizer="adam", loss="mse")


model.fit(x_train, y_train, epochs=epochs)

print(y_pred)
print(model.predict(y_pred))
model.save("Xor_Problem.h5")
