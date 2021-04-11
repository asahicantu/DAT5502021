from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer




def SBX(shape,n_classes):
    model =  Sequential()
    model.add(layers.Conv2D(32, kernel_size=(1, 1), activation='relu', input_shape=shape))
    model.add(layers.Conv2D(64, kernel_size=(1, 1), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(1, 1)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(n_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model