from librosa.core import audio
from librosa.filters import mel
import numpy as np
import librosa
import os
'''Extracting spectrogram features from wave files.

Uses librosa, reads a wave audio file.
Returns a three-value tuple variable called out
Such varialbe has:
out[0] Mel Spectrogram
out[1] MFCC (Mel Frecuency Cepstral coefficients)
out[2] Constant-Q transform of an audio signal
out[3] Wave file sample rate
out[4] Time in seconds of each array element
All returned spectrogram have a normalized wave amplitude of 0-1 decibels
'''

FEATURE_TYPES = ('cqt', 'mel', 'mfcc')
#%%
class Feature:
    DB_RANGE = 80.0
    def __init__(self,
                midi_time_step,
                midi_notes,
                file_path,
                features_to_process,
                sample_rate= 16000,
                max_freq=8500,
                verbose=False):
        try:
            self.Verbose = verbose
            self.Name = os.path.split(os.path.splitext(file_path)[0])[-1]
            self.Midi_Notes = midi_notes
            self.Midi_Time_Step = midi_time_step
            self.Total_Midi_time = self.Midi_Time_Step * len(self.Midi_Notes)
            
            data = Feature._get_wav_features(   file_path, 
                                                sample_rate,
                                                max_freq,
                                                features_to_process,
                                                verbose=self.Verbose)
            self.Sample_rate = data[0]
            self.Audio_Time = data[1]
            self.Data = data[2]
            
            self.Total_Audio_Time = self.Audio_Time[-1]

            self.Midi_2_Audio_Time = self._parse_audio_sample_to_midi_time()
        except (SystemError, FileNotFoundError) as e:
            raise

    def _parse_audio_sample_to_midi_time(self):
        audio_shape  = self.Data[list(self.Data.keys())[0]].shape[1]
        midi_delta = self.Midi_Time_Step
        audio_delta =  self.Total_Audio_Time/audio_shape
        midi_time = []
        midi_2_audio_time = []
        time_step = 0
        for time_step_idx in range(audio_shape):
            midi_time.append(time_step_idx)
            time_step += audio_delta
            #print(time_step, midi_delta )
            if time_step > midi_delta or time_step > self.Total_Midi_time:
                midi_delta += self.Midi_Time_Step
                midi_2_audio_time.append(midi_time.copy())
                midi_time = []
                if time_step >  self.Total_Midi_time:
                    # Time exceeded indication.
                    # At this point MIDI file contains less data in time 
                    # Than normal wave data
                    #print(f'Time exceeded {time_step}, {self.Total_Midi_time}')
                    break
        return midi_2_audio_time
    
    @staticmethod
    def _get_wav_features(filepath, sample_rate, max_freq,features_to_process, verbose=False):
        out = dict()
        try:
            sample_rate = sample_rate
            n_mels = 200
            n_mfcc = 200
            n_bins = 200
            audio_data, sample_rate = librosa.load(filepath, sr = sample_rate)
            time = np.arange(0,len(audio_data))/sample_rate

            for f2p in features_to_process:
                if verbose:
                    print(f'Extracing {f2p}')
                data = None
                if f2p == 'mel':
                    data = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=n_mels, fmax=max_freq)
                    data = librosa.power_to_db(data, ref=np.max, top_db=Feature.DB_RANGE)
                elif f2p == 'mfcc':
                    data = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mels=n_mfcc, n_mfcc=n_mfcc, fmax=max_freq)
                    data = librosa.power_to_db(data, ref=np.max, top_db=Feature.DB_RANGE)
                elif f2p == 'cqt':
                    data = librosa.cqt(y=audio_data, sr=sample_rate, bins_per_octave=12*2, fmin=librosa.note_to_hz('C0'), n_bins=n_bins)
                    data = librosa.amplitude_to_db(np.abs(data), ref=np.max, top_db=Feature.DB_RANGE)
                data =  Feature._normalize(data)
                out[f2p] = data
            out = (sample_rate, time, out)
            return out
            
        except FileNotFoundError:
            raise FileNotFoundError(f'File {filepath} could not be found')
        except Exception as e:
            raise SystemError(f'Internal Librosa error occurred {e}')
    
    # scale a spectrogram of a relative db range to [0, 1] where 1 is the peak volume
    @staticmethod
    def _normalize(C):
        return (C - np.min(C))/(np.max(C)-np.min(C))