import tensorflow as tf
import numpy as np
from Utils import  Misc
from tqdm import tqdm

class DataContainer(tf.keras.utils.Sequence):
    def __init__(self, data, labels, n_classes,sr,max_freq,batch_size=32,shuffle=True):
        self.data = data
        self.labels = labels
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sr = sr
        self.max_freq = max_freq
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # generate a batch of time data
        #X =  np.array( [Misc.vec2img(x, self.max_freq, self.sr) for x in self.data[indexes]])
        X =  self.data[indexes]
        Y = self.labels[indexes]
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)
