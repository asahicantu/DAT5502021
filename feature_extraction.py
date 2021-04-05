import numpy as np
import librosa

DB_RANGE = 80.0
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
def get_wav_features(filepath,verbose=False):
    try:
        audio_data, sample_rate = librosa.load(filepath)
        time = np.arange(0,len(audio_data))/sample_rate

        if verbose:
            print('Extracing MelSpec')
        melspec =  wav_to_melspec(audio_data, sample_rate)

        if verbose:
            print('Extracing mfcc')
        mfcc = wav_to_mfcc(audio_data, sample_rate)

        if verbose:
            print('Extracing cq')
        cq = wav_to_cq(audio_data, sample_rate)

        out = ( melspec , mfcc, cq,sample_rate,time)
        return out
    except FileNotFoundError:
        print(f'File {filepath} could not be found')
    except:
        print(f'Internal Librosa error occurred')



def wav_to_melspec(audio_data, sample_rate, n_mels=128, min_freq=27.5 ,max_freq=20000):
    S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=n_mels, fmax=max_freq)
    S_dB = librosa.power_to_db(S, ref=np.max, top_db=DB_RANGE)
    return normalize(S_dB)

def wav_to_mfcc(audio_data, sample_rate, n_mfcc=128, min_freq=27.5, max_freq=20000):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc, fmax=max_freq)
    mfcc_dB = librosa.power_to_db(mfcc, ref=np.max, top_db=DB_RANGE)
    return normalize(mfcc_dB)

def wav_to_cq(audio_data, sample_rate, n_bins=88, min_freq=27.5):
    cqt = librosa.cqt(y=audio_data, sr=sample_rate, fmin=min_freq, n_bins=n_bins)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max, top_db=DB_RANGE)
    return normalize(cqt_db)
# scale a spectrogram of a relative db range to [0, 1] where 1 is the peak volume
def normalize(C):
    return (C - np.min(C))/(np.max(C)-np.min(C))

    


