#%%
import os

data_path = os.path.join('..','data','split_midi') 
mid_file = str(os.path.join(data_path,'song1._7.mid'))
MIDI_SOUND_FONT = 'C:\\development\\repos\\Uni\\dat550-2021\\DeepWhiners\\198_Yamaha_SY1_piano.sf2'
MIDI_SOUND_FONT = 'C:\\MIDI\\SoundFonts\\MuseScore_General.sf2'
# The default tempo is 120 BPM.
# (500000 microseconds per beat (quarter note).)
DEFAULT_TEMPO = 500000
DEFAULT_TICKS_PER_BEAT = 480
 # Tempo is given in microseconds per beat (default 500000).
    # At this tempo there are (500000 / 1000000) == 0.5 seconds
    # per beat. At the default resolution of 480 ticks per beat
    # this is:
    #
    #    (500000 / 1000000) / 480 == 0.5 / 480 == 0.0010417
    #
    #return (tempo / 1000000.0) / ticks_per_beat
#%%
a = 0

#%%

import os
import librosa
import mido
import time
from mido import MidiFile
from mido.messages.messages import Message
from mido import midifiles
import sys
#https://mido.readthedocs.io/en/latest/midi_files.html
def play_midi(mid, max_steps = sys.maxsize,tempo = 500000):
    # The default tempo is 120 BPM.
    # (500000 microseconds per beat (quarter note).)
    DEFAULT_TEMPO = 500000
    DEFAULT_TICKS_PER_BEAT = 480
    split_interval = 1/64 # Representing minimum standard duration in musical notation 
    mid.tracks[0][3].tempo = DEFAULT_TEMPO * 4
    with  mido.open_output('Microsoft GS Wavetable Synth 0') as output :
        t0 = time.time()
        step = 0
        
        #for message in mid:
        for message in mid.play(meta_messages=False):
            if message.is_meta:
                continue
            print("step " + str(step) + " "  + str(message))
            output.send(message)
            step += 1
            if step > max_steps:
                break
        print('play time: {:.2f} s (expected {:.2f})'.format(time.time() - t0, mid.length))


def split_midi(file_path, chunks_per_second=1):
    chunks = dict()
    if os.path.exists(file_path):
        mid = MidiFile(file_path, clip=True,debug=False)
        duration = 0.0
        chunk_idx = 0
        for msg in mid:
            duration += msg.time
            if msg.type == 'note_on':
                pass
    #mid.print_tracks()
    #print(mid.ticks_per_beat)
    #print(mid.length)
    return chunks, mid
m_data,mid = split_midi(r"TestData\sample.mid")     
   
#play_midi(mid,tempo=2000000)
remove = [x for x in enumerate()]
#%%


#%%
mid.tracks[0][20].time = 1000
mid.tracks[0][20]
#%%


# %%
