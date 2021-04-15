import tensorflow
import numpy as np
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import tensorflow as tf
import os
from Utils.Models.CNN1D import CNN1D
from Utils.Models.CNN2D import CNN2D
from Utils.Models.LSTM import LSTM
from Utils.Models.SBX import SBX
from Utils.Models.DataContainer import DataContainer
import matplotlib.pyplot as plt

def train(X_train, y_train, X_test, y_test,batch_size,n_classes,model_type,sr,max_freq,shape):
    models = {'CNN1D':CNN1D,
              'CNN2D':CNN2D,
              'LSTM': LSTM,
              'SBX': SBX}

    assert model_type in models.keys(), '{} not an available model'.format(model_type)

    model = models[model_type](shape,n_classes)
    cp = ModelCheckpoint('models/{}.h5'.format(model_type), monitor='val_loss',
                         save_best_only=True, save_weights_only=False,
                         mode='auto', save_freq='epoch', verbose=3)
    csv_path = os.path.join('Data','Out','Log', '{}_log.csv'.format(model_type))
    early_stopping = EarlyStopping()
    csv_logger = CSVLogger(csv_path, append=False)


    tc = DataContainer(X_train, y_train,  n_classes, sr, max_freq, batch_size=batch_size)
    vc = DataContainer(X_test, y_test,  n_classes, sr, max_freq, batch_size=batch_size)

    model.fit(tc, validation_data=(vc),epochs=30,batch_size = batch_size, verbose=3,callbacks=[csv_logger,cp,early_stopping])
    return model


def plot_metric(model, metric):
    train_metrics = model.history[metric]
    val_metrics = model.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()
