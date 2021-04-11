import tensorflow
import numpy as np
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
import tensorflow as tf
import os
from Utils.Models.CNN1D import CNN1D
from Utils.Models.CNN2D import CNN2D
from Utils.Models.LSTM import LSTM
from Utils.Models.SBX import SBX
from Utils.Models.DataContainer import DataContainer


def train(X_train, y_train, X_test, y_test,batch_size,n_classes,model_type):
    models = {'CNN1D':CNN1D,
              'CNN2D':CNN2D,
              'LSTM': LSTM,
              'SBX': SBX}

    assert model_type in models.keys(), '{} not an available model'.format(model_type)


    # xt_shape = X_train.shape
    # yt_shape = y_train.shape
    # #X_train = X_train.reshape(xt_shape[0],xt_shape[1]*xt_shape[2],1)
    # y_train = y_train.reshape(yt_shape[0],1,yt_shape[1])

    # xt_shape = X_test.shape
    # yt_shape = y_test.shape
    # #X_test = X_test.reshape(xt_shape[0],xt_shape[1]*xt_shape[2],1)
    # y_test = y_test.reshape(yt_shape[0],1,yt_shape[1])


    tg = DataContainer(X_train, y_train, n_classes, batch_size=batch_size)
    vg = DataContainer(X_test, y_test, n_classes, batch_size=batch_size)

    X_train = np.array([spec.flatten() for spec in X_train])
    y_train = np.array(y_train)

    X_test = np.array([spec.flatten() for spec in X_test])
    y_test = np.array(y_test)

    shape =  X_train[0].shape


    model = models[model_type](shape,n_classes)
    

    cp = ModelCheckpoint('models/{}.h5'.format(model_type), monitor='val_loss',
                         save_best_only=True, save_weights_only=False,
                         mode='auto', save_freq='epoch', verbose=3)
    csv_path = os.path.join('Data','Out','Log', '{}_log.csv'.format(model_type))

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))


    csv_logger = CSVLogger(csv_path, append=False)
    model.fit(X_train,y_train, validation_data=(X_test,y_test),epochs=30,batch_size = batch_size, verbose=3,callbacks=[csv_logger, cp])