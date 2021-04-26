#%%
 # Tempo is given in microseconds per beat (default 500000).
    # At this tempo there are (500000 / 1000000) == 0.5 seconds
    # per beat. At the default resolution of 480 ticks per beat
    # this is:
    #
    #    (500000 / 1000000) / 480 == 0.5 / 480 == 0.0010417
    #
    #return (tempo / 1000000.0) / ticks_per_beat

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
import glob
import tqdm
from midi2audio import FluidSynth
MIDI_SOUND_FONT = "Soundfont/198_Yamaha_SY1_piano.sf2"


#https://mido.readthedocs.io/en/latest/midi_files.html

DEFAULT_TEMPO = 500000 # BITS PER SECOND = 0.5 SECONDS PER BEAT
DEFAULT_TICKS_PER_BEAT = 480 # 4
NOTE_32 = 1/32
NOTE_16 = 1/16
NOTES_8 = 1/8
MIDI_NOTES = 88 # Starting from A0 to C8
MIDI_OFFSET = 21 # Note A0 starts at MIDI value 21. Required an offset to match properly
# The default tempo is 120 BPM.
# (500000 microseconds per beat (quarter note).)

def get_midi_file(midi_file,tempo = 1/8):
    if os.path.exists(midi_file):
        mid = load_midi(midi_file)     
        midi_split = split_midi(mid,verbose=False,note_factor = tempo)
        return midi_split
    else :
        raise SystemError(f"Path {directory_path} does not exist")


def get_midi_files(directory_path,max_elements = sys.maxsize,tempo = 1/16, verbose = False):
    files = dict()
    note_factor = tempo
    if os.path.exists(directory_path):
        for i, file in tqdm.tqdm(enumerate( glob.glob(directory_path + r'\*mid*',recursive=True) ),desc='Process Midi..'):
            if i > max_elements -1:
                break
            f = os.path.basename(file)
            mid = load_midi(file)     
            midi_split = split_midi(mid,verbose=verbose,note_factor = note_factor)
            files[f] =  (note_factor, midi_split)
        return files
    else :
        raise SystemError(f"Path {directory_path} does not exist")

def play_midi(mid, max_steps = sys.maxsize,tempo = 500000):
    # The default tempo is 120 BPM.
    # (500000 microseconds per beat (quarter note).)
    DEFAULT_TEMPO = 500000
    DEFAULT_TICKS_PER_BEAT = 480
    NOTE_32 = 1/32
    NOTE_16 = 1/16
    
    split_interval = NOTE_32 # Representing minimum standard duration in musical notation 
    with  mido.open_output('Microsoft GS Wavetable Synth 0') as output :
        step = 0
        t0 = time.time()
        for message in mid.play(meta_messages=False):
            try:
                print("step " + str(step) , message.type, str(message))
                if message.is_meta \
                or message.type =='control_change' \
                or message.type =='program_change' \
                or message.type == 'sysex':
                    continue
                
                output.send(message)
                step += 1

                if step > max_steps:
                    break
            except:
                print(message.type)
        print('play time: {:.2f} s (expected {:.2f})'.format(time.time() - t0, mid.length))

def load_midi(file_path, chunks_per_second=1):
    chunks = dict()
    if os.path.exists(file_path):
        return MidiFile(file_path, clip=True,debug=False)
    #mid.print_tracks()
    #print(mid.ticks_per_beat)
    #print(mid.length)
    return None

def split_midi(mid, max_steps = sys.maxsize,tempo = 500000,note_factor = NOTE_16,verbose = False):
    note_events = ['note_on','note_off','set_tempo']
    notes = np.zeros(MIDI_NOTES,dtype=int)
    time_notes = []
    note_step = 0
    delta_time = 0
    for message in mid:
        if message.type  not in note_events:
            continue
        if verbose:
            print(f'step {note_step} {message}')
        
        delta_time += message.time
        while delta_time > note_factor :
            #note_val = ''.join([str(x) for x in np.copy(notes).tolist()])
            note_val = notes.tolist()
            time_notes.append(note_val)
            delta_time -= note_factor
        
        if 'note' in message.type:
            note_idx = message.note - MIDI_OFFSET
            if note_idx  >= 0  and  note_idx < MIDI_NOTES:
                notes[note_idx ] = 1 if message.type == 'note_on' and message.velocity >  0 else 0
        
        if note_step > max_steps: # Break the loop and exit
            break
        note_step +=1
    while delta_time > 0 :
        #note_val = ''.join([str(x) for x in notes.tolist()])
        note_val = notes.tolist()
        time_notes.append(note_val)
        delta_time -= note_factor
    return time_notes

def vec_to_mid(mid_vector,tempo = DEFAULT_TEMPO, ticks = DEFAULT_TICKS_PER_BEAT):
    messages = []
    time_delta = 0
    note_cache = np.zeros(MIDI_NOTES,dtype=int)
    for vec in mid_vector:
        for i,note in enumerate(vec[1]) :
            if note_cache[i] != note:
                note_cache[i] = note
                message = None
                time = mido.midifiles.second2tick(time_delta,ticks,tempo)
                time = int(time)
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

# %%
def midi2wave(midi_filepath, out_filepath, sound_font=MIDI_SOUND_FONT, sample_rate=44100):
    target_dir = ''.join(os.path.split(midi_filepath)[:-1])
    if os.path.exists(target_dir):
        fs = FluidSynth(sound_font=sound_font, sample_rate=sample_rate)
        fs.midi_to_audio(midi_filepath, out_filepath)
    else:
        raise IOError(f'Directory {target_dir} does not exist')
