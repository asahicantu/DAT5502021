import numpy as np
import librosa

def wav_to_melspec(filepath, n_mels=128, min_freq=27.5 ,max_freq=20000):
    audio_data, sample_rate = librosa.load(filepath)
    S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=n_mels, fmax=max_freq)
    S_dB = librosa.power_to_db(S, ref=np.max)
        
    return S_dB, sample_rate

def wav_to_mfcc(filepath, n_mfcc=20, min_freq=27.5, max_freq=20000):
    audio_data, sample_rate = librosa.load(filepath)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc, fmax=max_freq)
    mfcc_dB = librosa.power_to_db(mfcc, ref=np.max)
        
    return mfcc_dB, sample_rate

def wav_to_cq(filepath, n_bins=88, min_freq=27.5):
    audio_data, sample_rate = librosa.load(filepath)
    cqt = librosa.cqt(y=audio_data, sr=sample_rate, fmin=min_freq, n_bins=88)
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    
    return cqt_db, sample_rate