import tensorflow
import numpy as np
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import tensorflow as tf
import os
import Utils.Models.CNN1D as CNN1D
import Utils.Models.CNN2D as CNN2D
import Utils.Models.LSTM as LSTM
import Utils.Models.SBX as SBX
from Utils.Models.DataContainer import DataContainer
import matplotlib.pyplot as plt
import importlib
importlib.reload(CNN1D)
importlib.reload(CNN2D)
importlib.reload(LSTM)
importlib.reload(SBX)

def train(X_train, y_train, X_test, y_test,batch_size,n_classes,model_type,sr,max_freq,shape):
    models = {'CNN1D':CNN1D.CNN1D,
              'CNN2D':CNN2D.CNN2D,
              'LSTM': LSTM.LSTM,
              'SBX': SBX.SBX}

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

    model.fit(X_train,y_train, validation_data=(X_test,y_test),epochs=30,batch_size = batch_size, verbose=3,callbacks=[csv_logger,cp,early_stopping])
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
