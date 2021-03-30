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
from mido import MidiTrack
from mido.messages.messages import Message
from mido import midifiles
import sys
import numpy as np
#https://mido.readthedocs.io/en/latest/midi_files.html

DEFAULT_TEMPO = 500000 # BITS PER SECOND = 0.5 SECONDS PER BEAT
DEFAULT_TICKS_PER_BEAT = 480 # 4
NOTE_32 = 1/32
NOTE_16 = 1/16
NOTES_8 = 1/8
MIDI_NOTES = 87 # Starting from A0 to C8
MIDI_OFFSET = 21 # Note A0 starts at MIDI value 21. Required an offset to match properly
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

        while delta_time > note_factor :
            #note_val = ''.join([str(x) for x in np.copy(notes).tolist()])
            note_val = notes.tolist()
            time_notes.append((round(note_factor,4), note_val))
            delta_time -= note_factor
        if message.type in note_events:
            notes[note_idx ] = 1 if message.type == 'note_on' and message.velocity >  0 else 0
        if note_step > max_steps:
            break
        note_step +=1
    while delta_time > 0 :
        #note_val = ''.join([str(x) for x in notes.tolist()])
        note_val = notes.tolist()
        time_notes.append((round(delta_time,4), note_val))
        delta_time -= note_factor
    return time_notes



def vec_to_mid(mid_vector,tempo = DEFAULT_TEMPO, ticks = DEFAULT_TICKS_PER_BEAT):
    messages = []
    time_delta = 0
    note_cache = np.zeros(MIDI_NOTES,dtype=int)
    time_delta = 0
    for vec in mid_vector:
        for i,note in enumerate(vec[1]) :
            if note_cache[i] != note:
                note_cache[i] = note
                message = None
                time = mido.midifiles.second2tick(time_delta,ticks,tempo)
                if note == 0:
                    message = Message(type="note_off",note= MIDI_OFFSET + i,velocity=0,time=  time )
                if note == 1:
                    message = Message(type="note_on",note= MIDI_OFFSET + i,velocity=127,time= time)
                messages.append(message)
                time_delta = 0
        time_delta += vec[0]
    return messages

def get_mid_tempo(mid,track_idx=0):
    final_tempo = DEFAULT_TEMPO
    for msg in mid.tracks[track_idx]:
        if msg.is_meta and msg.type=="set_tempo":
            final_tempo = msg.tempo
    return final_tempo

def recreate_mid(mid_msgs):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    for msg in mid_msgs:
        track.append(msg)
    return mid

def pretty_print_vector(vec):
    for v in vec:
        print(v[0], ''.join([ str(x) for x in v[1] ]),sep='-')
#%%            
def test():
    #mid = split_midi(r"..\data\archive\data\undertale\Undertale - Small Shock.mid")     
    #mid = load_midi(r"..\data\archive\data\undertale\Undertale - Oh My.mid")     
    #mid = load_midi(r"TestData\\sample.mid")     
    mid = load_midi(r"..\data\archive\data\anime\Suiheisen.mid")     
    tempo = get_mid_tempo(mid,track_idx = 0)
    #play_midi(mid)
    vec = split_midi(mid)
    
    #msgs = mid_msgs = vec_to_mid(vec,tempo = tempo)
    #mid = recreate_mid(msgs)
    #for 
    #mid.print_tracks()
    #play_midi(mid)
    #mid.print_tracks()


# %%
