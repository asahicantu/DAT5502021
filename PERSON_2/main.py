import mido
from lossfunc import midiloss

mid1 = mido.MidiFile('test.mid', clip=True)
mid2 = mido.MidiFile('birthday.mid', clip=True)

#checking
print(midiloss(mid1, mid2))
print(midiloss(mid1, mid1))
print(midiloss(mid2, mid2))