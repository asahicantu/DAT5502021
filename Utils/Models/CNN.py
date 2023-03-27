import tensorflow as tf
import keras
from tensorflow.keras import layers, optimizers, backend
from tensorflow.keras.layers import TimeDistributed, LayerNormalization, Conv2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.engine.base_layer import Layer
from keras.layers import Input, Flatten,Conv2D,Dense, MaxPool2D,Dropout,MaxPooling2D
from keras.models import Model
from . Accuracy import AccuracyHistory, root_mse,r2_coeff_determination
def CNN1D(feature, shape,n_classes=88):
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
    model = Model(inputs=base_layer.input, outputs=o, name='feature')
    model.compile(optimizer='adam',
                  loss='sigmoid',
                  metrics=['accuracy'])
    return model

def CNN2D(feature, shape,n_classes):
    """ Creates a model"""
    model =  Sequential(name = feature)
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
    model.add(Dense(n_classes, activation='softmax'))
    adam = optimizers.Adam(learning_rate=0.0001, decay=.00001)
    model.compile(loss=root_mse,
                  optimizer=adam,
                  metrics=[root_mse, 'mae',r2_coeff_determination])
    return model

def ALEXNET(feature,shape,n_classes):
    model = Sequential(name = f'{feature}_ALEXNET')
    model.add( Conv2D(filters=96, kernel_size=(3,3), strides=(2,2), activation='relu', input_shape=shape))
    model.add( BatchNormalization())
    model.add( MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add( Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
    model.add( BatchNormalization())
    model.add( MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add( Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add( BatchNormalization())
    model.add( Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add( BatchNormalization())
    model.add( Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add( BatchNormalization())
    model.add( MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add( Flatten())
    model.add( Dense(4096, activation='relu'))
    model.add( Dropout(0.5))
    model.add( Dense(4096, activation='relu'))
    model.add( Dropout(0.5))
    model.add( Dense(n_classes, activation='sigmoid'))
    model.compile(loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(learning_rate=.0001, decay=1e-6),
            metrics=['accuracy'])
    return model
    

def CNN2D_V2(feature, shape,n_classes):
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
    model.add(Dense(n_classes, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(learning_rate=.0001, decay=1e-6),
            metrics=['accuracy'])
    return model    



def CNN2D_CUSTOM(feature,shape, n_classes):
    model = Sequential(name = f'{feature}_CNN2D_CUSTOM')
    model.add( layers.experimental.preprocessing.Rescaling(1, input_shape=shape))
    model.add( layers.Conv2D(16, 3, padding='same', activation='relu'))
    model.add( layers.MaxPooling2D())
    model.add( layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add( layers.MaxPooling2D())
    model.add( layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add( layers.MaxPooling2D())
    model.add( layers.Flatten())
    model.add( layers.Dense(128, activation='relu'))
    model.add( layers.Dense(n_classes))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


def CNN2D_V3(feature,shape, n_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1),activation='tanh',input_shape=shape))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(64, (3,3), activation='tanh'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Conv2D(64, (5,5), activation='relu'))
    # Final output layer
    #model.add(Conv2D(128, (5,5), activation='sigmoid'))
    #model.add(Flatten())
    model.add(Flatten())
    #model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(n_classes, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(lr=.0001, decay=1e-6),
            metrics=['accuracy'])

    return model