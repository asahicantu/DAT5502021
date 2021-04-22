import tensorflow as tf


def LSTM(feature, shape, N_CLASSES=88):
    """
    LSTM Neural Network
    """
    model = tf.keras.Sequential(name=feature)
    model.add(tf.keras.layers.LSTM(units=200, activation='tanh', return_sequences=True, input_shape = shape))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.LSTM(units=200, activation='tanh', return_sequences=True))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.LSTM(units=200, activation='tanh', return_sequences=False))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(88, activation='relu'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])



    return model


def LSTM_M2M(feature, shape, N_CLASSES=88):
    """
    LSTM Time distributed   
    """

    model = tf.keras.Sequential(name = feature)
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

