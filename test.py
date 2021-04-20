#%%
import sys
import os
import numpy as np
#from tqdm.notebook import trange, tqdm
from tqdm  import tqdm
import matplotlib.pyplot as plt
from Utils import Misc, Pickle, Preprocess, TrainTestValid, Evaluation
from Utils.Models import Train
import datetime
import tensorflow as tf
import gc
import argparse
import time

def init():
  print('init...')
  tf.config.list_physical_devices('GPU')

  gpus = tf.config.list_physical_devices('GPU')
  os.environ["CUDA_VISIBLE_DEVICES"] = f"{len(gpus)}"
  logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

  
def runModel(feature,
              model_type,
              log_path,
              x,
              y,
              batch_size,
              epochs,
              gpu):
  #tf.debugging.set_log_device_placement(True)
  X,Y = TrainTestValid.train_test_validation_split(x,y)
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
    x_valid = X['valid']
    y_valid = Y['valid']
    
    
    num_classes = y.shape[1]
    shape = x_test.shape[1:]
       
    model = Train.train(
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
    prediction = model(x_valid)
    cm = None
    try:
      cm =Evaluation.get_confusion_matrix(prediction, y_valid)
    except:
      print('Failed to generate confusion matrix')
    return model, prediction, cm

def trainFeatures(
  features,
  features_img_out,
  y,
  pickle_path,
  log_path,
  gpu,
  batch_size,
  epochs
  ):
  models = {x:dict() for x in features_img_out.keys()}
  cms = {x:dict() for x in features_img_out.keys()}
  predictions = {x:dict() for x in features_img_out.keys()}
  model_types = Train.CNN_MODELS
  for key in features_img_out.keys():
    x = features_img_out[key]
    x = x.astype('float64') / 255    
    
    gc.collect()
    for model_type in model_types:
      try:
        pickle_file = os.path.join(pickle_path,f'fit_model_{key}_{model_type}.pkl')
        if os.path.exists(pickle_file):
          print(f'Pickle file {pickle_file} exists. Skipping...')
        else:
          print(f'Processing model {key}:{model_type}')
          model, prediction, cm = runModel( key,model_type,log_path, x,y,batch_size,epochs,gpu) 
          models[key][model_type] = model
          predictions[key][model_type] = prediction
          cms[key][model_type] = cm
          Pickle.dump_pickle(pickle_file,(model,prediction,cm))
        gc.collect()
      except Exception as e:
        print(f'Model {key} {model_type} failed at {e}')



def parse_args():
  '''Parse arguments'''
  data_source_path = Misc.get_dir(os.getcwd(),'data','in','classical') #MIDI dataset
  out_path =  Misc.get_dir('Data','Out')
  out_wav_path = Misc.get_dir('Data','Out','Wav')
  pickle_path = Misc.get_dir('Data','Out','Pickle')
  img_path = Misc.get_dir('Data','Out','Img')
  log_path = Misc.get_dir('Logs','Fit')

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('data_source_path', type=str, nargs='?', default=data_source_path, help='MIDI dataset path')
  parser.add_argument('out_path', type=str, nargs='?', default=out_path, help ='existing folder on the disk for output data')
  parser.add_argument('out_wav_path', type=str, nargs='?', default=out_wav_path, help ='path where wave files will be saved')
  parser.add_argument('pickle_path', type=str, nargs='?', default=pickle_path, help='directory where pickle data will be saved')
  parser.add_argument('img_path', type=str, nargs='?', default=img_path, help='directory where image data will be stored')
  parser.add_argument('log_path', type=str, nargs='?', default=log_path, help='directory where log data will be saved during ML trainig')
  parser.add_argument('max_sample_size', type=int, nargs='?', const=sys.maxsize,default=sys.maxsize, help='maximum number of elements to sample')
  parser.add_argument('max_freq', type=int, nargs='?', const=8500,default=8500, help='Maximum frequency to process audio files')
  parser.add_argument('tempo', type=int, nargs='?', const=1/8,default=1/8, help='Tempo at which midi files will be split')
  parser.add_argument('sample_rate', type=int, nargs='?', const=16000,default=16000, help='Sample rate to process wafe files')
  parser.add_argument('scale_img', type=int, nargs='?', const=6, default=6, help='Scale at which processed spectrogram images will be scaled upon Initial size=(432 x 648)')
  parser.add_argument('max_images', type=int, nargs='?', const=sys.maxsize, default=sys.maxsize, help='Maximum number of samples to convert into image spectrograms')
  parser.add_argument('img_num_save', type=int, nargs='?', const=1000,default=1000, help='If image path specified, maximum number of images to save when running spectrogram')
  parser.add_argument('batch_size', type=int, nargs='?', const=8, default=32, help='Batch size to run when training the model')
  parser.add_argument('epochs', type=int, nargs='?', const=100,default=100, help='Number of epochs to run when training the model')
  parser.add_argument('gpu', type=int, nargs='?', const=0,default=0, help='If available the GPU unit to run ML algorithms')
  return parser.parse_args()


def main():
  start = time.time()
  args = parse_args()
  
  init()

  (features, features_img_out,y)   = Preprocess.preprocess(
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
  gc.collect()

  trainFeatures(
    features, 
    features_img_out, 
    y,
    args.pickle_path,
    args.log_path,
    args.gpu,
    args.batch_size,
    args.epochs)
  
  
  end = time.time()
  print(f'Total time: f{end - start}')
    
  #%%
  # print(cm)
  #   Evaluation.plot_roc_curve(predictions, y_valid)
  #   Train.plot_metric(model.history, 'loss')
  #   Train.plot_metric(model.history, 'acc')
  # #%%
#%%
if __name__ == '__main__':
  main()


