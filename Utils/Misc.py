import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import librosa.display
from PIL import Image
import os

def rgb2gray(rgb):
  return np.dot(rgb[...,:3],[0.2989,0.5870,0.1140])


def vec2img(vec,fmax,sr,scale,img_type):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    img = None
    if img_type == 'mel':
        img = librosa.display.specshow(vec, y_axis='mel',x_axis='s', fmax=fmax,sr = sr,ax=ax)
    elif img_type == 'cqt':
        img = librosa.display.specshow(vec, y_axis='cqt_hz', x_axis='s', fmax=fmax,sr = sr,ax=ax)
    else:
        img = librosa.display.specshow(vec, y_axis='log',x_axis='s', fmax=fmax,sr = sr,ax=ax)
    #fig.tight_layout(pad=1.0)
    canvas.draw()
    (w,h) = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb() , dtype='uint8' )
    buf.shape = ( w, h,3 )
    w = int(w/scale)
    h = int(h/scale)
    size = (80,106)
    img = Image.fromarray(buf).resize(size,Image.ANTIALIAS)
    buf =  np.asarray(img)
    return buf

def create_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_dir(*arg):
    final_dir = ''
    for a in arg:
        final_dir = os.path.join(final_dir,a)
        create_dir_if_not_exist(final_dir)
    return final_dir


