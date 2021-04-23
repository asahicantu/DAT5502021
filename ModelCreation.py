#%%
import sys
import os
import numpy as np
#from tqdm.notebook import trange, tqdm
from tqdm  import tqdm
import matplotlib.pyplot as plt
from Utils import Misc, Pickle, Preprocess, TrainTestValid, Evaluation
from Utils.Models import Model
import datetime
import tensorflow as tf
import gc
import argparse
import time

from Utils.TrainTestValid import split_in_sequences


def init():
  print('init...')
  tf.config.list_physical_devices('GPU')

  gpus = tf.config.list_physical_devices('GPU')
  os.environ["CUDA_VISIBLE_DEVICES"] = f"{len(gpus)}"
  logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  for gpu in gpus:
    print(gpu)

  
def runModel(feature,
              model_type,
              log_path,
              X,
              Y,
              batch_size,
              epochs,
              gpu):
  #tf.debugging.set_log_device_placement(True)
  
  #strategy = tf.distribute.MirroredStrategy(["/device:GPU:0", "/device:GPU:1"])
  #with strategy.scope():
  with tf.device(f'/device:GPU:{gpu}'):
    #config = tf.config.set_logical_device_configuration()
    #gpu_options = tf. .GPUOptions(allow_growth=True)
    #session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    x_train = tf.convert_to_tensor(X['train'])
    y_train = tf.convert_to_tensor(Y['train'])
    x_test = tf.convert_to_tensor(X['test'])
    y_test = tf.convert_to_tensor(Y['test'])

    # proper formatting
    if model_type == 'LSTM':
      x_train = tf.transpose(x_train, [0, 2, 1])
      x_test = tf.transpose(x_test, [0, 2, 1])
    elif model_type == 'MLP' or model_type == 'MLP_1H' or model_type == 'LSTM_M2M':
      x_train = np.array([tf.reshape(x, [-1]) for x in x_train])
      x_test = np.array([tf.reshape(x, [-1]) for x in x_test])
      if model_type == 'LSTM_M2M':
        x_train = np.array(split_in_sequences(x_train, 10, keep_short_batch=False))
        x_test = np.array(split_in_sequences(x_test, 10, keep_short_batch=False))
        y_train = np.array(split_in_sequences(y_train, 10, keep_short_batch=False))
        y_test = np.array(split_in_sequences(y_test, 10, keep_short_batch=False))
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
  
    num_classes = y_test.shape[1]
    shape = x_test.shape[1:]
        
    model = Model.train(
      log_path,
      feature,
      x_train,
      y_train,
      x_test,
      y_test,
      batch_size,
      epochs,
      num_classes,
      model_type,
      shape)
    
    return model

def trainFeatures(
  models_types,
  features,
  y,
  pickle_path,
  model_path,
  log_path,
  gpu,
  batch_size,
  epochs
  ):
  for feature in features.keys():
    x = features[feature]
    X = None
    Y = None
    gc.collect()
    for model_type in models_types:
      try:
        model_file = os.path.join(model_path, f'{model_type}_{feature}.h5')
        if os.path.exists(model_file):
          print(f'Model {model_file} exists. Skipping...')
        else:
          print(f'Processing model {feature}:{model_type}')
          if X is None and Y  is None:
             X, Y = TrainTestValid.train_test_validation_split(x,y)
          model = runModel( feature,model_type, log_path, X,Y,batch_size,epochs,gpu)
          model.save(model_file)
        gc.collect()
      except Exception as e:
        print(f'Model {feature} {model_type} failed at {e}')

def parse_args():
  '''Parse arguments'''
  data_source_path = Misc.get_dir(os.getcwd(),'data','in','classical') #MIDI dataset
  out_path =  Misc.get_dir('Data','Out')
  out_wav_path = Misc.get_dir('Data','Out','Wav')
  pickle_path = Misc.get_dir('Data','Out','Pickle')
  img_path = Misc.get_dir('Data','Out','Img')
  log_path = Misc.get_dir('Logs','Fit')
  model_path = Misc.get_dir('Data','Out','Model')

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('data_source_path', type=str, nargs='?', default=data_source_path, help='MIDI dataset path')
  parser.add_argument('out_path', type=str, nargs='?', default=out_path, help ='existing folder on the disk for output data')
  parser.add_argument('out_wav_path', type=str, nargs='?', default=out_wav_path, help ='path where wave files will be saved')
  parser.add_argument('pickle_path', type=str, nargs='?', default=pickle_path, help='directory where pickle data will be saved')
  parser.add_argument('img_path', type=str, nargs='?', default=img_path, help='directory where image data will be stored')
  parser.add_argument('model_path', type=str, nargs='?', default=model_path, help='directory where model data will be saved during ML trainig')
  parser.add_argument('log_path', type=str, nargs='?', default=log_path, help='directory where log data will be saved during ML trainig')
  parser.add_argument('max_sample_size', type=int, nargs='?', const=sys.maxsize,default=sys.maxsize, help='maximum number of elements to sample')
  parser.add_argument('max_freq', type=int, nargs='?', const=8500,default=8500, help='Maximum frequency to process audio files')
  parser.add_argument('tempo', type=int, nargs='?', const=1/8,default=1/8, help='Tempo at which midi files will be split')
  parser.add_argument('sample_rate', type=int, nargs='?', const=16000,default=16000, help='Sample rate to process wafe files')
  parser.add_argument('scale_img', type=int, nargs='?', const=6, default=6, help='Scale at which processed spectrogram images will be scaled upon Initial size=(432 x 648)')
  parser.add_argument('max_images', type=int, nargs='?', const=sys.maxsize, default=sys.maxsize, help='Maximum number of samples to convert into image spectrograms')
  parser.add_argument('img_num_save', type=int, nargs='?', const=0,default=0, help='If image path specified, maximum number of images to save when running spectrogram')
  parser.add_argument('batch_size', type=int, nargs='?', const=32, default=32, help='Batch size to run when training the model')
  parser.add_argument('epochs', type=int, nargs='?', const=100,default=100, help='Number of epochs to run when training the model')
  parser.add_argument('gpu', type=int, nargs='?', const=0,default=0, help='If available the GPU unit to run ML algorithms')
  return parser.parse_args()


def main():
  start = time.time()
  args = parse_args()
  
  init()
  (features_1D, features_2D,y)   = Preprocess.preprocess(
                        data_source_path = args.data_source_path,
                        out_wav_path = args.out_wav_path ,
                        img_path = args.img_path,
                        pickle_path = args.pickle_path,
                        max_sample_size=args.max_sample_size,
                        max_freq = args.max_freq,
                        tempo = args.tempo,
                        sample_rate = args.sample_rate,
                        img_num_save= args.img_num_save,
                        scale_img = args.scale_img,
                        max_images = sys.maxsize,
                        )

  y = y.astype('float64')    
  features_2D = {x[0]:np.array( x[1], dtype = 'float64')/255 for x in features_2D.items() }
  gc.collect()
  print('Training models for 1-D Features....')
  trainFeatures(Model.MODELS_1D,features_1D,y,args.pickle_path,args.model_path,args.log_path,args.gpu,args.batch_size,args.epochs)
  gc.collect()
  print('Training models for 2-D Features....')
  trainFeatures(Model.MODELS_2D,features_2D,y,args.pickle_path,args.model_path,args.log_path,args.gpu,args.batch_size,args.epochs)
  end = time.time()
  diff = end - start
  print(f'PRocess Completed in {diff} seconds')
  #%%
  # print(cm)
  #   Evaluation.plot_roc_curve(predictions, y_valid)
  #   Train.plot_metric(model.history, 'loss')
  #   Train.plot_metric(model.history, 'acc')
  # #%%
#%%
if __name__ == '__main__':
  main()


