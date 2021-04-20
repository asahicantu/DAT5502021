import tensorflow as tf


def LSTM_TD(feature, shape, N_CLASSES=88):

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
