import tensorflow as tf
import keras
from tensorflow.keras import layers, optimizers, backend
from tensorflow.keras.layers import TimeDistributed, LayerNormalization, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.engine.base_layer import Layer
from keras.layers import Input, Flatten,Conv2D,Dense, MaxPool2D,Dropout,MaxPooling2D
from keras.models import Model

def CNN1D(shape,n_classes=88):
    base_layer = layers.Input(shape=shape,name='Input_Layer')
    x = LayerNormalization(axis=2, name='batch_norm')(base_layer)
    x = TimeDistributed(layers.Conv1D(8, kernel_size=(4), activation='tanh'), name='td_conv_1d_tanh')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_1')(x)
    x = TimeDistributed(layers.Conv1D(16, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_1')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_2')(x)
    x = TimeDistributed(layers.Conv1D(32, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_2')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_3')(x)
    x = TimeDistributed(layers.Conv1D(64, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_3')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_4')(x)
    x = TimeDistributed(layers.Conv1D(128, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_4')(x)
    x = layers.GlobalMaxPooling2D(name='global_max_pooling_2d')(x)
    x = layers.Dropout(rate=0.1, name='dropout')(x)
    x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
    o = layers.Dense(n_classes, activation='softmax', name='softmax')(x)
    model = Model(inputs=base_layer.input, outputs=o, name='1d_convolution')
    model.compile(optimizer='adam',
                  loss='sigmoid',
                  metrics=['accuracy'])
    return model

def CNN2D(shape,N_CLASSES):
    """ Creates a model"""
    model =  tf.keras.Sequential()
    model.add(Conv2D(filters=2, kernel_size=(1, 2), strides=(1),    padding='same', activation='relu', input_shape=shape))
    model.add(Conv2D(filters=2, kernel_size=(7, 1), strides=(1),    padding='same', activation='relu'))
    model.add(Conv2D(filters=3, kernel_size=(1, 2), strides=(1),    padding='same', activation='relu'))
    model.add(Conv2D(filters=3, kernel_size=(7, 1), strides=(1),    padding='same', activation='relu'))
    model.add(Conv2D(filters=4, kernel_size=(1, 2), strides=(1, 2), padding='same', activation='relu'))
    model.add(Conv2D(filters=4, kernel_size=(1, 2), strides=(1, 2), padding='same', activation='relu'))
    model.add(Conv2D(filters=5, kernel_size=(1, 2), strides=(1, 2), padding='same', activation='relu'))
    model.add(Conv2D(filters=5, kernel_size=(1, 2), strides=(1, 2), padding='same', activation='relu'))
    model.add(Conv2D(filters=5, kernel_size=(1, 2), strides=(1, 2), padding='same', activation='relu'))
    model.add(Conv2D(filters=6, kernel_size=(1, 2), strides=(1),    padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(N_CLASSES, activation='softmax'))
    model.summary()
    adam = optimizers.Adam(lr=0.0001, decay=.00001)
    model.compile(loss=root_mse,
                  optimizer=adam,
                  metrics=[root_mse, 'mae', r2_coeff_determination])
    return model

def ALEXNET(shape,n_classes):
    model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=shape),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

def CNN2D_V2(shape,n_classes):
    model = Sequential(name=feature)
    model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1),activation='tanh',input_shape=shape))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(256, (3,3), activation='tanh'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Conv2D(64, (5,5), activation='relu'))
    # Final output layer
    #model.add(Conv2D(128, (5,5), activation='sigmoid'))
    #model.add(Flatten())
    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(lr=.0001, decay=1e-6),
            metrics=['accuracy'])



def CNN2D_CUSTOM(shape, n_classes):
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1, input_shape=shape),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_classes)
    ])
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


