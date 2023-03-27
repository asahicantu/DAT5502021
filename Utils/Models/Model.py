import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from Utils import Misc
import Utils.Models.CNN as CNN
import Utils.Models.LSTM as LSTM
import Utils.Models.MLP as MLP
from Utils.Models import Accuracy
import matplotlib.pyplot as plt
import importlib
import datetime


MODELS = {'CNN1D': CNN.CNN1D,
          'CNN2D': CNN.CNN2D,
          'ALEXNET': CNN.ALEXNET,
          'CNN2D_V2': CNN.CNN2D_V2,
          'CNN2D_V3': CNN.CNN2D_V3,
          'CNN2D_CUSTOM': CNN.CNN2D_CUSTOM,
          'LSTM': LSTM.LSTM,
          'LSTM_M2M': LSTM.LSTM_M2M,
          'MLP': MLP.MLP,
          'MLP_1H': MLP.MLP_1_hidden,
          }

MODELS_2D = [
    'CNN2D',
    'ALEXNET',
    'CNN2D_V2',
    'CNN2D_V3',
    'CNN2D_CUSTOM'
]

MODELS_1D = [
    'LSTM',
    'LSTM_M2M',
    'MLP',
    'MLP_1H'
]



def train(log_path,feature, X_train, y_train, X_test, y_test, batch_size, epochs, n_classes, model_type, shape):
    assert model_type in MODELS.keys(), '{} not an available model'.format(model_type)

    model = MODELS[model_type](feature, shape, n_classes)

    model_ckpt = os.path.join('Data', 'Out', 'Model_Checkpoint', f'{model_type}_{feature}_ckpt.h5')
    history = Accuracy.AccuracyHistory()

    checkpoint = ModelCheckpoint(
        model_ckpt, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(
        patience=5, monitor='val_loss', verbose=1, mode='min')

    csv_path = os.path.join(log_path, f'{feature}_{model_type}_log.csv')
    csv_logger = CSVLogger(csv_path, append=False)
    log_dir = Misc.get_dir(log_path,f'{feature}_{model_type}')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [checkpoint, early_stop, csv_logger, history,tensorboard_callback]
    model.summary()
    print(f'Training with batch size: {batch_size} for {epochs} epochs...')
    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test),
            callbacks=callbacks)

    return model
