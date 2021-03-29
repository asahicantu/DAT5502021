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



import os
import librosa
import mido
import time
from mido import MidiFile
from mido.messages.messages import Message
#https://mido.readthedocs.io/en/latest/midi_files.html
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
    return chunks, mid
        
#m_data,mid = split_midi(r"C:\Users\asahi\Dev\Data\classical\bach\bach_846.mid")
m_data,mid = split_midi(r"C:\Users\asahi\Dev\Data\aa.mid")


for msg in mid:
    print(msg)  


print(mid.ticks_per_beat)
print(mid.length)


#mid2.print_tracks()
#type(mid2.tracks[0][0].is_meta)
#dir(mid2.tracks[0][0])


for i,m in enumerate(mid.tracks[0]):
    if m.type=="note_on" and m.velocity == 0:
        del mid.tracks[0][i]
    i +=-1

tm = 100
for m in mid.tracks[0]:
    if m.type == "note_on" and m.velocity > 0 :
        m.time = tm
        tm += 10
    else:
        m.time =m.time
        
    
    
mid.print_tracks()



### from mido.messages import Message
import time
with  mido.open_output('Microsoft GS Wavetable Synth 0') as output :
    t0 = time.time()
    max_play = 9
    step = 0
    for message in mid.play():
        if message.type == "note_on" and message.note ==  64:
            pass
            #message.note = 50
            #message = Message(type = "note_on",note=40, velocity = 127,time=message.time)
        #message.time=0 if message.time == 0 else 0.1
        output.send(message)
        
        print("step " + str(step) + " "  + str(message))
        #m = Message(type = "note_on",note=70, velocity = 100,time=0)
        m = Message(type = "note_on",note=77, velocity = 100,time=0.5)
        output.send(m)
        if step < max_play:
            step += 1
        else:
            break
    
    print(m)
    print('play time: {:.2f} s (expected {:.2f})'.format(
            time.time() - t0, mid.length))




    o = mido.open_output('Microsoft GS Wavetable Synth 0')


    o.send(Message(type="control_change", channel=0, control=121, value=0, time=0))
o.send(Message(type="program_change",channel=0, program= 0, time=0))
o.send(Message(type="control_change",channel=0, control= 7, time=0))
o.send(Message(type="control_change",channel=0, control= 10, time=0))
o.send(Message(type="control_change",channel=0, control= 91, time=0))
o.send(Message(type="control_change",channel=0, control= 93, time=0))
o.send(Message(type="note_on",channel=0, note= 93, velocity=0, time=0))
o.send(Message(type="note_on",channel=0, note= 93, velocity=100, time=1.89))
o.send(Message(type="note_on",channel=0, note= 93, velocity=60, time=1.89))



o.close()
o = None


import librosa
from mido import MidiFile
from mido.messages.messages import Message

notes = []
for msg in mid.tracks[0]:
    if msg.type == "note_on": 
        notes.append(msg.note)
#print(notes)
librosa.midi_to_note(notes, octave=True, cents=False, key='C:maj', unicode=True)