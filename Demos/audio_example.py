## From https://www.youtube.com/watch?v=TJGlxdW7Fb4
#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sounddevice as sd
import librosa
#%%
filename = librosa.ex('trumpet')
y, sr = librosa.load(filename)
time = np.arange(0,len(y))/sr
time[-10:]

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
y, sr = librosa.load('../Data/In/classical/Bartok_SZ080-02_002_20110315-SMD.mid')
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

# %%
from Utils import Midi
from  Utils import Feature
import importlib
import glob
import os
import numpy as np
import tqdm
importlib.reload(Midi)
importlib.reload(Feature)
MIDI_SOUND_FONT = 'Soundfont/198_Yamaha_SY1_piano.sf2'
# This path should point to the used midi dataset
data_source_path = r"D:\Dev\datasets\midi\anime"
# This path has to be an existing folder on the disk, otherwise the files will not be created in the following
out_wav_path = r"D:\Dev\datasets\midi\out"
# Sample rate in Hz, common CD quality is 44100
SAMPLE_RATE = 11025
mid_files_dict = Midi.get_midi_files(data_source_path,max_elements=1, verbose=False)
features = list()
keys =  list(mid_files_dict.keys())  # Avoid exception when deleting keys
for key in tqdm.tqdm(keys):
    try:
        in_file = os.path.join(data_source_path, key)
        out_file = key.replace('.mid', '.wav')
        out_file = os.path.join(out_wav_path, out_file)
        Midi.midi2wave(in_file,out_file, sample_rate=SAMPLE_RATE)
        midi_time = mid_files_dict[key][0]
        midi_data = mid_files_dict[key][1]
        feature = Feature.Feature(midi_time, midi_data, out_file)
        features.append(feature)
    except (SystemError, FileNotFoundError) as e:
        ## Ignore files whose feature extraction has failed
        print(e)    
f = features[0]
#%%
from mido import MidiFile
from mido import MidiTrack
from mido.messages.messages import Message
from mido import midifiles
from midi2audio import FluidSynth
MIDI_SOUND_FONT = "../Soundfont/198_Yamaha_SY1_piano.sf2"

