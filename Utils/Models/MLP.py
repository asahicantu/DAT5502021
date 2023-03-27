import tensorflow as tf


def MLP_1_hidden(feature, shape, N_CLASSES=88):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=shape),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(88, activation='sigmoid'),
    ], name=feature)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def MLP(feature, shape, N_CLASSES=88):
    """
    Multi layer perceptron
    """
    model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=shape),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(88, activation='sigmoid'),
    ], name=feature)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

