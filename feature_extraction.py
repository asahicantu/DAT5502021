import numpy as np
import librosa

DB_RANGE = 80.0

def wav_to_melspec(filepath, n_mels=128, min_freq=27.5 ,max_freq=20000):
    audio_data, sample_rate = librosa.load(filepath)
    S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=n_mels, fmax=max_freq)
    S_dB = librosa.power_to_db(S, ref=np.max, top_db=DB_RANGE)
        
    return normalize(S_dB), sample_rate

def wav_to_mfcc(filepath, n_mfcc=128, min_freq=27.5, max_freq=20000):
    audio_data, sample_rate = librosa.load(filepath)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc, fmax=max_freq)
    mfcc_dB = librosa.power_to_db(mfcc, ref=np.max, top_db=DB_RANGE)
        
    return normalize(mfcc_dB), sample_rate

def wav_to_cq(filepath, n_bins=88, min_freq=27.5):
    audio_data, sample_rate = librosa.load(filepath)
    cqt = librosa.cqt(y=audio_data, sr=sample_rate, fmin=min_freq, n_bins=n_bins)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max, top_db=DB_RANGE)
    
    return normalize(cqt_db), sample_rate

# scale a spectrogram of a relative db range to [0, 1] where 1 is the peak volume
def normalize(C):
    return (C - np.min(C))/(np.max(C)-np.min(C))