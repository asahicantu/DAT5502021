import numpy as np
from . import Feature

def smooth_predictions(predictions, window_length):
    
    smoothed_pred = np.zeros(predictions.shape)
    
    for i in range(predictions.shape[0] - window_length + 1):
        win = predictions[i : i + window_length]
        smoothed_pred[i] = np.round(np.average(win, axis=0))
    
    # smooth remaining values
    win = predictions[-window_length:]
    for i in range(predictions.shape[0] - window_length + 1, predictions.shape[0]):    
        smoothed_pred[i] = np.round(np.average(win, axis=0))
            
    return smoothed_pred

def foo(midi_file):
    tempo = 1/8
    midi_song = Midi.get_midi_file(midi_file,tempo=tempo)
    midi_path = os.path.split(midi_file)
    Midi.midi2wave(midi_file,f'./{midi_file}.wav',sound_font='../Soundfont/198_Yamaha_SY1_piano.sf2')
    features_to_process = Feature.FEATURE_TYPES
    MLP_MFCC_THRESHOLD = 0.052892983
    features_1D = {x:[] for x in features_to_process }
    feature = Feature.Feature(midi_song[0],
                            midi_song[1],
                            f'./{fp}.wav',
                            features_to_process,
                            sample_rate = 16000,
                            max_freq=8500,
                            verbose=False)
    y =[]
    max_len = np.max([len(i) for i in feature.Midi_2_Audio_Time])
    for index, timeframe in enumerate(feature.Midi_2_Audio_Time):
        y.append(feature.Midi_Notes[index])
        if len(timeframe) < max_len:
            t = timeframe.copy()
            for i in range(max_len-len(timeframe)):
                t.append(timeframe[-i])
                for f2p in features_to_process:
                    features_1D['mfcc'].append(feature.Data[f2p][:,t])
        else:
            for f2p in features_to_process:
                features_1D[f2p].append(feature.Data[f2p][:,timeframe])
    y = np.array(y)
    X = np.array(features_1D['mfcc'])
    X = np.array([x.flatten() for x in X])



