import tensorflow as tf
from tensorflow.keras import layers, optimizers, backend
from tensorflow.keras.layers import TimeDistributed, LayerNormalization, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.engine.base_layer import Layer

def CNN2D(shape,N_CLASSES):
    ##def create_model(input_height, input_width, one_d_array_len):
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

def root_mse(y_true, y_pred):
    # returns tensorflow.python.framework.ops.Tensor
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))

def r2_coeff_determination(y_true, y_pred):
    SS_res = backend.sum(backend.square(y_true - y_pred))
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
    # epsilon avoids division by zero
    return (1 - SS_res / (SS_tot + backend.epsilon()))


def CNN2DD(shape, N_CLASSES):
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
        layers.Dense(N_CLASSES)
    ])
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              
              metrics=['accuracy'])
    return model


    base_layer = layers.Input(shape=shape)
    x = layers.Conv2D(16, 2, padding='same', activation='relu')(base_layer),
    x = layers.MaxPooling2D(pool_size=(2,2),padding='same')(x),
    x = layers.Conv2D(32, 2, padding='same', activation='relu')(x),
    x = layers.MaxPooling2D()(x),
    x = layers.Conv2D(64, 2, padding='same', activation='relu')(x),
    x = layers.MaxPooling2D()(x),
    x = layers.Flatten()(x),
    x = layers.Dense(128, activation='relu')(x),
    o = layers.Dense(N_CLASSES)(x)

    # x = LayerNormalization(axis=2, name='batch_norm')(base_layer)
    # x = layers.Conv2D(16, kernel_size=(2,2), activation='tanh', padding='same', name='conv2d_tanh')(x)
    # x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_3')(x)
    # x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_3')(x)
    # x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_4')(x)
    # x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_4')(x)
    # x = layers.Flatten(name='flatten')(x)
    # x = layers.Dense(128, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
    # o = layers.Dense(N_CLASSES, activation='softmax', name='softmax')(x)

    model = Model(inputs=base_layer, outputs=o, name='2d_convolution')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model



