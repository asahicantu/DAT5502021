#%%
import os
import numpy as np
from tqdm.notebook import trange, tqdm
import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
from Utils import Misc, Pickle, Preprocess, TrainTestValid, Evaluation
from Utils.Models import Train

import tensorflow as tf
import keras
from keras.layers import Input, Flatten,Conv2D,Dense, MaxPool2D,Dropout,MaxPooling2D
from keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras import backend
import numpy as np
import datetime
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

tf.config.list_physical_devices('GPU')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class AccuracyHistory(keras.callbacks.Callback):
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


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

def rgb2gray(rgb):
  return np.dot(rgb[...,:3],[0.2989,0.5870,0.1140])

# This path should point to the used midi dataset
data_source_path = os.path.join(os.getcwd(),'data','in','classical')

# This path has to be an existing folder on the disk, otherwise the files will not be created in the following
out_path =  Misc.get_dir('Data','Out')
out_wav_path = Misc.get_dir('Data','Out','wav')
pickle_path = Misc.get_dir('Data','Out','Pickle')
img_path = Misc.get_dir('Data','Out','Img')

(features_out, features_img_out,y)   = Preprocess.preprocess(data_source_path,
                      out_wav_path,
                      img_path,
                      pickle_path,
                      max_sample_size=sys.maxsize,
                      max_freq = 8500,  #20000
                      tempo = 1/8,
                      sample_rate = 16000, # Sample rate in Hz, common CD quality is 44100
                      img_num_save= 1000,
                      scale_img = 4,
                      max_images = sys.maxsize,
                      )
#%%
def runModel(feature,
              x,
              y,
              batch_size=64,
              epochs=100):
  x = x.astype('float64') / 255    
  y = y.astype('float64')    
  X,Y = TrainTestValid.train_test_validation_split(x,y)

  
  log_dir = "logs/fit/" + feature + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  tf.debugging.set_log_device_placement(True)
  devices = [x[0] for x in tf.config.experimental.list_logical_devices('GPU')]
  model_ckpt = os.path.join('Data','Out','Model',f'{feature}_ckpt.h5')
  #strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0'])
    #with strategy.scope():
  with tf.device('/device:GPU:0'):
      x_train = tf.convert_to_tensor(X['train'])
      y_train = tf.convert_to_tensor(Y['train'])
      x_test = tf.convert_to_tensor(X['test'])
      y_test = tf.convert_to_tensor(Y['test'])
      x_valid = X['valid']
      y_valid = Y['valid']

      num_classes = y.shape[1]
      shape = x_test.shape[1:]

      model = Sequential(name=feature)
      model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1),activation='tanh',input_shape=shape))
      model.add(Dropout(0.5))
      model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
      model.add(Conv2D(256, (3,3), activation='tanh'))
      model.add(Dropout(0.5))
      model.add(MaxPooling2D(pool_size=(2,2)))
      #model.add(Conv2D(64, (5,5), activation='relu'))
      # Final output layer
      #model.add(Conv2D(128, (5,5), activation='sigmoid'))
      #model.add(Flatten())
      model.add(Flatten())
      model.add(Dense(64, activation='sigmoid'))
      model.add(Dense(num_classes, activation='sigmoid'))

      model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(lr=.0001, decay=1e-6),
              metrics=['accuracy'])

      history = AccuracyHistory()

      checkpoint = ModelCheckpoint(model_ckpt,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
      early_stop = EarlyStopping(patience=5, monitor='val_loss',verbose=1, mode='min')
      callbacks = [history, checkpoint,early_stop]

      model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=callbacks)

      prediction = model(x_valid)
      cm = Evaluation.get_confusion_matrix(prediction, y_valid)
      return model, prediction, cm
    

#%%
models = dict()
cms = dict()
predictions = dict()
for key in features_out.keys():
  x = features_img_out[key]
  model, prediction, cm = runModel(key, x,y,batch_size=32)  
  models[key] = model
  predictions[key] = prediction
  cms[key] = cm
#%%

print(cm)
  Evaluation.plot_roc_curve(predictions, y_valid)
  Train.plot_metric(model.history, 'loss')
  Train.plot_metric(model.history, 'acc')
#%%




