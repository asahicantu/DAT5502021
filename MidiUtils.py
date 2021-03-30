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
import numpy as np
#https://mido.readthedocs.io/en/latest/midi_files.html

DEFAULT_TEMPO = 500000
DEFAULT_TICKS_PER_BEAT = 480
NOTE_32 = 1/32
NOTE_16 = 1/16
# The default tempo is 120 BPM.
# (500000 microseconds per beat (quarter note).)

def play_midi(mid, max_steps = sys.maxsize,tempo = 500000):
    # The default tempo is 120 BPM.
    # (500000 microseconds per beat (quarter note).)
    DEFAULT_TEMPO = 500000
    DEFAULT_TICKS_PER_BEAT = 480
    NOTE_32 = 1/32
    NOTE_16 = 1/16
    
    split_interval = 1/32 # Representing minimum standard duration in musical notation 
    with  mido.open_output('Microsoft GS Wavetable Synth 0') as output :
        step = 0
        t0 = time.time()
        for message in mid.play(meta_messages=False):
            if message.is_meta \
            or message.type =='control_change' \
            or message.type =='program_change':
                continue
            print("step " + str(step) + " "  + str(message))
            output.send(message)
            step += 1

            if step > max_steps:
                break
        print('play time: {:.2f} s (expected {:.2f})'.format(time.time() - t0, mid.length))


def load_midi(file_path, chunks_per_second=1):
    chunks = dict()
    if os.path.exists(file_path):
        return MidiFile(file_path, clip=True,debug=False)
    #mid.print_tracks()
    #print(mid.ticks_per_beat)
    #print(mid.length)
    return None

def split_midi(mid, max_steps = sys.maxsize,tempo = 500000,note_factor = NOTE_16):
    MIDI_NOTES = 87 # Starting from A0 to C8
    MIDI_OFFSET = 21 # Note A0 starts at MIDI value 21. Required an offset to match properly
    note_factor = NOTE_32
    note_events = ['note_on','note_off']
    notes = np.zeros(MIDI_NOTES,dtype=int)
    time_notes = []
    note_step = 0
    delta_time = 0
    for message in mid:
        if message.is_meta \
        or message.type =='control_change' \
        or message.type =='program_change':
            continue
        print(f'step {note_step} {message}')
        note_idx   = message.note - MIDI_OFFSET
        delta_time += message.time
        
        if delta_time / note_factor > 1:
            delta_time -= note_factor
            note_val = ''.join([str(x) for x in np.copy(notes).tolist()])
            time_notes.append( (round(note_factor,4), note_val) )
            notes = np.array(notes,copy = True)
        if message.type in note_events:
            notes[note_idx ] = 1 if message.type == 'note_on' and message.velocity >  0 else 0
        if note_step > max_steps:
            break
        # if message.time > 0:
        #     note_val = ''.join([str(x) for x in np.copy(notes).tolist()])
        #     time_notes.append((round(message.time,4), note_val))
        #     #notes = np.array(notes,copy = True)
        note_step +=1
    while delta_time > 0 :
        note_val = ''.join([str(x) for x in notes.tolist()])
        time_notes.append((round(note_factor,4), note_val))
        delta_time -= note_factor
        #note_val = ''.join([str(x) for x in np.zeros(MIDI_NOTES,dtype=int).tolist()])
    
    return time_notes
#mid = split_midi(r"..\data\archive\data\undertale\Undertale - Small Shock.mid")     
mid = load_midi(r"..\data\archive\data\undertale\Undertale - Oh My.mid")     
a = split_midi(mid)
#m_data,mid = play_midi(r"TestData\sample.mid")     
#play_midi(mid,max_steps=7)
print(len(a))
a
#%%
from MIDI import MIDIFile
from sys import argv

def parse(file):
    c=MIDIFile(file)
    c.parse()
    print(str(c))
    for idx, track in enumerate(c):
        track.parse()
        print(f'Track {idx}:')
        print(str(track))
parse(r"..\data\archive\data\undertale\Undertale - Oh My.mid")     
#%%
mid.tracks[0][20].time = 1000
mid.tracks[0][20]
#%%


# %%
