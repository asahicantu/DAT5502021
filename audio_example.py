## From https://www.youtube.com/watch?v=TJGlxdW7Fb4
#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sounddevice as sd
import librosa

# %%
mpl.rcParams['figure.figsize'] = [12,8]
mpl.rcParams.update({'font.size' : 18})

dt = 0.001
t = np.arange(0,2,dt)
f0 = 0
f1 = 900
t1 = 2
x = np.cos(2 * np.pi * t * ( f0 + (f1 - f0) * np.power(t,2)/(3 * t1 ** 2) ) )
fs = 1/dt
sd.play(2*x, fs)

plt.specgram(x, NFFT=128, Fs=1/dt, noverlap=120, cmap='jet_r')
plt.colorbar()
plt.show()
# %%
import librosa
import librosa.display
import numpy as np
y, sr = librosa.load(librosa.ex('trumpet'))
librosa.feature.melspectrogram(y=y, sr=sr)
D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D, sr=sr)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
# %%
