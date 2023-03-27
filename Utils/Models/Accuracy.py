import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.callbacks import Callback
class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

def root_mse(y_true, y_pred):
    # returns tensorflow.python.framework.ops.Tensor
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))

def r2_coeff_determination(y_true, y_pred):
    SS_res = backend.sum(backend.square(y_true - y_pred))
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
    # epsilon avoids division by zero
    return (1 - SS_res / (SS_tot + backend.epsilon()))
