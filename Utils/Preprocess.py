import os
import numpy as np
from tqdm  import tqdm
#from tqdm.notebook import trange, tqdm
import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
import importlib
import glob
from Utils import Misc, Midi, Feature, Pickle
from PIL import Image


def preprocess(
        data_source_path,
        out_wav_path,
        img_path,
        pickle_path,
        max_sample_size = sys.maxsize,
        max_freq = 8500,  #20000
        tempo = 1/16,
        sample_rate = 16000, # Sample rate in Hz, common CD quality is 44100
        img_num_save= 10,
        scale_img = 6, 
        max_images = 1000
        ):


    
    features_1D_path = os.path.join(pickle_path, 'features_1D.pkl')
    features_2D_path = os.path.join(pickle_path, 'features_2D.pkl')
    labels_path = os.path.join(pickle_path, 'labels.pkl')
    
    features_1D = Pickle.load_pickle(features_1D_path)
    features_2D = Pickle.load_pickle(features_2D_path)
    
    y = Pickle.load_pickle(labels_path ) 

    features_to_process = ['mel','mfcc','cqt']
    if features_1D is None and y is None:
        print('Pickle data not found, running pre processing...')
    
        mid_files_dict = Midi.get_midi_files(data_source_path,max_elements =max_sample_size, tempo = tempo, verbose=False)
        features = []

        wavs = [x for x in os.listdir(out_wav_path) if x.endswith('.wav')]

        for key in tqdm(mid_files_dict.keys(),desc='Feature Extract...'):
            try:
                in_file = os.path.join(data_source_path, key)
                out_file = key.replace('.mid', '.wav')
                out_file = os.path.join(out_wav_path, out_file)
                if out_file not in wavs:
                    Midi.midi2wave(in_file,out_file, sample_rate=sample_rate)
                midi_time = mid_files_dict[key][0]
                midi_data = mid_files_dict[key][1]
                feature = Feature.Feature(  midi_time,
                                            midi_data,
                                            out_file,
                                            features_to_process,
                                            sample_rate = sample_rate,
                                            max_freq=max_freq,
                                            verbose=False)
                features.append(feature)
            except (SystemError, FileNotFoundError) as e:
                ## Ignore files whose feature extraction has failed
                print(e)

        y = []
        features_1D = {x:[] for x in features_to_process }
        features_2D = {x:[] for x in features_to_process }

        max_len = np.max([len(i) for f in features for i in f.Midi_2_Audio_Time])
        for feature in tqdm(features, desc='Feature Merge...'):
            for index, timeframe in enumerate(feature.Midi_2_Audio_Time):
                y.append(feature.Midi_Notes[index])
                
                # padding for smaller chunks by duplicating the last slice of a spectrogram
                if len(timeframe) < max_len:
                    t = timeframe.copy()
                    for i in range(max_len-len(timeframe)):
                        t.append(timeframe[-i])
                        for f2p in features_to_process:
                            features_1D[f2p].append(feature.Data[f2p][:,t])
                else:
                    for f2p in features_to_process:
                        features_1D[f2p].append(feature.Data[f2p][:,timeframe])
        y = np.array(y)
        
        Pickle.dump_pickle(features_1D_path ,features_1D)
        Pickle.dump_pickle(labels_path,y)

    
    if features_2D is None:
        features_2D=dict()
        print('Pickle image data not found, running pre processing...')
        for f2p in tqdm(features_to_process, desc='Vec2Image...'):
                data = np.array(features_1D[f2p])[ : max_images]
                print(data.shape)
                features_2D[f2p] =  np.array(
                    [ Misc.vec2img(x,max_freq,sample_rate,scale_img,f2p) for x in tqdm(data) ] )
        Pickle.dump_pickle(features_2D_path,features_2D)


    if not img_path is None:
        print('Saving images..')     
        for f2p in tqdm(features_to_process):
            Misc.get_dir(img_path,f2p)

        if img_num_save > 0:
            for f2p in tqdm(features_to_process,desc='Img Save...'): 
                data = features_2D[f2p]
                [Image.fromarray(x).save(os.path.join(img_path, f2p, f"TRAIN_{i}.jpeg")) for i,x in tqdm(enumerate(data[:img_num_save]))]    
    return (features_1D, features_2D,y)    


def main():
    preprocess()

if __name__ == '__main__':
    main()