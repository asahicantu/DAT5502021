import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from Utils import Misc
import Utils.Models.CNN as CNN
import Utils.Models.LSTM as LSTM
import Utils.Models.Accuracy
import matplotlib.pyplot as plt
import importlib

def train(feature,X_train, y_train, X_test, y_test,batch_size,epochs,n_classes,model_type,shape):
    models = {  'CNN1D'         :CNN.CNN1D,
                'CNN2D'         :CNN.CNN2D,
                'ALEXNET'       :CNN.ALEXNET,    
                'CNN2D_V2'      :CNN.CNN2D_V2,    
                'CNN2D_CUSTOM'  :CNN.CNN2D_CUSTOM,    
                'LSTM'          : LSTM.LSTM
                }

    assert model_type in models.keys(), '{} not an available model'.format(model_type)

    model = models[model_type](feature,shape, n_classes)

    model_ckpt = os.path.join('Data','Out','Model_Checkpoint',f'{feature}_{model_type}_ckpt.h5')
    history =  Utils.Models.Accuracy.AccuracyHistory()

    checkpoint = ModelCheckpoint(model_ckpt,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
    early_stop = EarlyStopping(patience=5, monitor='val_loss',verbose=1, mode='min')

    csv_path =  Misc.get_dir('Data','Out','Log')
    csv_path = os.path.join(csv_path, f'{feature}_{model_type}_log.csv')
    csv_logger = CSVLogger(csv_path, append=False)
    callbacks = [checkpoint,early_stop,csv_logger,history]
    model.summary()
    print(f'Training with batch size: {batch_size} for {epochs} epochs...')
    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test),
            callbacks=callbacks)

    model_path = os.path.join('Data','Out','Model',f'{feature}_{model_type}.h5')    
    model.save(model_path)
    return model

