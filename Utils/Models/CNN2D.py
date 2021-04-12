from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer

# def CNN2DD():
#     input_layer = layers.Input(shape=(X.shape[1],))
#     dense_layer_1 = layers.Dense(15, activation='relu')(input_layer)
#     dense_layer_2 = layers.Dense(10, activation='relu')(dense_layer_1)
#     output = layers.Dense(y.shape[1], activation='softmax')(dense_layer_2)

#     model = Model(inputs=input_layer, outputs=output)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#     return model




def CNN2D(shape, N_CLASSES):
    #base_layer = layers.Flatten(input_shape=data_shape)
    base_layer = layers.Input(shape=shape)
    x = LayerNormalization(axis=2, name='batch_norm')(base_layer)
    x = layers.ZeroPadding2D(padding=(2, 2))(x)
    x = layers.Conv2D(8, kernel_size=(7,7), activation='tanh', padding='same', name='conv2d_tanh')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_1')(x)
    x = layers.ZeroPadding2D(padding=(2, 2))(x)
    x = layers.Conv2D(16, kernel_size=(5,5), activation='relu', padding='same', name='conv2d_relu_1')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_2')(x)
    x = layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_2')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_3')(x)
    x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_3')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_4')(x)
    x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_4')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(rate=0.2, name='dropout')(x)
    x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
    o = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)
    model = Model(inputs=base_layer, outputs=o, name='2d_convolution')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
