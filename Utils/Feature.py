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
#%%
class Feature:
    DB_RANGE = 80.0
    def __init__(self,midi_time_step, midi_notes, file_path, sample_rate= 16000,max_freq=8500, verbose=False):
        try:
            self.VERBOSE = verbose
            data = Feature._get_wav_features(file_path, sample_rate=sample_rate, max_freq=max_freq, verbose=self.VERBOSE)
            self.Name = os.path.split(os.path.splitext(file_path)[0])[-1]
            self.Midi_Notes = midi_notes
            self.Midi_Time_Step = midi_time_step
            self.Total_Midi_time = self.Midi_Time_Step * len(self.Midi_Notes)
            self.Sample_rate = data[0]
            self.Audio_Time = data[1]
            self.MelSpec = data[2]
            self.MFCC = data[3]
            self.Total_Audio_Time = self.Audio_Time[-1]
            self.CQ = data[4]
            self.Midi_2_Audio_Time = self._parse_audio_sample_to_midi_time()
        except (SystemError, FileNotFoundError) as e:
            raise

    def _parse_audio_sample_to_midi_time(self):
        midi_delta = self.Midi_Time_Step
        audio_delta =  self.Total_Audio_Time/self.MFCC.shape[1]
        midi_time = []
        midi_2_audio_time = []
        time_step = 0
        for time_step_idx in range(self.MFCC.shape[1]):
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
    def _get_wav_features(filepath, sample_rate, max_freq, verbose=False):
        try:
            sample_rate = sample_rate
            n_mels = 200
            n_mfcc = 200
            n_bins = 200

            audio_data, sample_rate = librosa.load(filepath, sr = sample_rate)
            time = np.arange(0,len(audio_data))/sample_rate

            if verbose:
                print('Extracing MelSpec')
            mel = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=n_mels, fmax=max_freq)
            mel = librosa.power_to_db(mel, ref=np.max, top_db=Feature.DB_RANGE)
            mel =  Feature._normalize(mel)

            if verbose:
                print('Extracing mfcc')
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mels=n_mfcc, n_mfcc=n_mfcc, fmax=max_freq)
            mfcc = librosa.power_to_db(mfcc, ref=np.max, top_db=Feature.DB_RANGE)
            mfcc =  Feature._normalize(mfcc)

            if verbose:
                print('Extracing cq')
            cq = librosa.cqt(y=audio_data, sr=sample_rate, bins_per_octave=12*2, fmin=librosa.note_to_hz('C0'), n_bins=n_bins)
            cq = librosa.amplitude_to_db(np.abs(cq), ref=np.max, top_db=Feature.DB_RANGE)
            cq = Feature._normalize(cq)

            out = (sample_rate, time, mel , mfcc, cq)
            return out
            
        except FileNotFoundError:
            raise FileNotFoundError(f'File {filepath} could not be found')
        except Exception as e:
            raise SystemError(f'Internal Librosa error occurred {e}')
    
    # scale a spectrogram of a relative db range to [0, 1] where 1 is the peak volume
    @staticmethod
    def _normalize(C):
        return (C - np.min(C))/(np.max(C)-np.min(C))