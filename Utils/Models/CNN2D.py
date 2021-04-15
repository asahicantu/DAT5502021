from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers 
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
    base_layer = layers.Input(shape=shape)
    

    x = LayerNormalization(axis=2, name='batch_norm')(base_layer)
    x = Conv2D(16, kernel_size=(3,3), activation='tanh', padding='same', name='conv2d_tanh')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_3')(x)
    x = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_3')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_4')(x)
    x = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_4')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
    o = Dense(N_CLASSES, activation='sigmoid', name='softmax')(x)

    model = Model(inputs=base_layer, outputs=o, name='2d_convolution')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model



def CNN2D_2(shape,N_CLASSES):
        model = Sequential()
        model.add(Conv2D(filters=2, kernel_size=(1, 2), strides=(1), padding='same', activation='relu', input_shape=shape))
        model.add(Conv2D(filters=2, kernel_size=(7, 1), strides=(1), padding='same', activation='relu'))
        model.add(Conv2D(filters=3, kernel_size=(1, 2), strides=(1), padding='same', activation='relu'))
        model.add(Conv2D(filters=3, kernel_size=(7, 1), strides=(1), padding='same', activation='relu'))
        for i in range(2):
            model.add(Conv2D(filters=4, kernel_size=(1, 2), strides=(1, 2), padding='same',
                            activation='relu'))
        for i in range(3):
            model.add(Conv2D(filters=5, kernel_size=(1, 2), strides=(1, 2), padding='same',
                            activation='relu'))
        model.add(Conv2D(filters=6, kernel_size=(1, 2), strides=(1), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(N_CLASSES, activation='sigmoid'))
        model.summary()
        adam = optimizers.adam(lr=0.0001, decay=.00001)
        model.compile(loss=root_mse,
                    optimizer=adam,
                    metrics=[root_mse, 'mae', r2_coeff_determination])
        return model

def root_mse(y_true, y_pred):
    # returns tensorflow.python.framework.ops.Tensor
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))

def r2_coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    # epsilon avoids division by zero
    return (1 - SS_res / (SS_tot + K.epsilon()))

