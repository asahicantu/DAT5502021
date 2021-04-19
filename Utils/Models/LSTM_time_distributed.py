from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import os

from tensorflow.python.keras.engine import base_layer

def LSTM_TD(shape, N_CLASSES=88):
    
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=200, activation='tanh', return_sequences=True, input_shape=shape))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.LSTM(units=200, activation='tanh', return_sequences=True))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.LSTM(units=200, activation='tanh', return_sequences=True))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(200, activation='relu')))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(88, activation='sigmoid')))
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
