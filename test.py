#%%
import sys
import os
import numpy as np
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
from Utils import Misc, Pickle, Preprocess, TrainTestValid, Evaluation
from Utils.Models import Train
import datetime
import tensorflow as tf


tf.config.list_physical_devices('GPU')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


# This path should point to the used midi dataset
data_source_path = os.path.join(os.getcwd(),'data','in','classical')
# This path has to be an existing folder on the disk, otherwise the files will not be created in the following
out_path =  Misc.get_dir('Data','Out')
out_wav_path = Misc.get_dir('Data','Out','wav')
pickle_path = Misc.get_dir('Data','Out','Pickle')
img_path = Misc.get_dir('Data','Out','Img')

(_, features_img_out,y)   = Preprocess.preprocess(
                      data_source_path = data_source_path,
                      out_wav_path = out_wav_path ,
                      img_path = None,
                      pickle_path = pickle_path,
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

      model, prediction, cm = Train.train(
        feature,
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size,
        num_classes,
        'ALEXNET',
        shape)
      return model, prediction, cm = 
    

#%%
models = dict()
cms = dict()
predictions = dict()
for key in features_img_out.keys():
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




