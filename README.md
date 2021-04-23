# DAT550-2021 Final Project

## Overview

Music Notation Generation

### Description:

Create neural network capable of "listening to music waveforms" and generate sheet musical notation equivalent in "abc" language.

#### Expected results:
- Train a deep neural network model able to provide given outputs from music waveforms with good accuracy

#### Goal:
To go from music waveforms to  abc format such as in https://piano2notes.com/file/7747aefc-c516-47b3-abdb-dbcda1649a65 (Does not generate abc but  music sheet)
Dataset:
http://theremin.music.uiowa.edu/MISpiano.html


### SoundFonts
https://github.com/FluidSynth/fluidsynth/wiki/SoundFont

## Algorithms
- Spectrogram
- MFCC Spectrogram
    https://en.wikipedia.org/wiki/Mel-frequency_cepstrum#:~:text=From%20Wikipedia%2C%20the%20free%20encyclopedia,nonlinear%20mel%20scale%20of%20frequency.

## Used Datasets
### [PIANO Notes Transcription from Kaggle](https://www.kaggle.com/arshadgeek/piano-notes-transcription)
This dataset is used by default to explore data frequencies
    

## Candidate datasets

- https://www.kaggle.com/juliancienfuegos/what-is-a-note
- https://www.kaggle.com/soumikrakshit/classical-music-midi
- https://magenta.tensorflow.org/datasets/maestro#v300

### MIDI FIle format specification
https://github.com/colxi/midi-parser-js/wiki/MIDI-File-Format-Specifications

## Kaggle Repository
https://www.kaggle.com/asahicantu/dat550-2021-final-project/

## Tasks
[] Try different preprocessing techniques and find out its differences


0. Transform Audio MIDI Files into waveforms
1. Iterate over best features to extract and process
2. Candidate algorithms to use
   1. Given an audio sample dataset, split a file into smaller time chunks
   2. Fourier transformation
   3. Frequency over time diagram
4. [] Get Audio Files Data
   1. Creating them  or get dataset
   2. 
5. [] Create feature extraction mechanism
6. [ ]
7. Report []
   1. Summary [TIM]
   2. Data Preprocessing [Asahi]
   3. Related work (Existing approaches) [Lucas]
   4. CNN [Asahi]
   5. LSTM
   6. FEED FORWARD
   7. Logistic Regression [] [Lucas]
   8. Conclusions
8. Code []
   1. Postprocessing
      1. Convert Prediction back to midi [MIDI_Notation] [ABC_Notation] [Music_Notation] [ASAHI]
   2. Smoothing techniques
   3. Early stopping [TIM]
   4. Logistic Regression [Lucas]
   
      



 