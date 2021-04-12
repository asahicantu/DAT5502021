import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import librosa.display
from PIL import Image

def vec2img(vec,fmax,sr,scale):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    img = librosa.display.specshow(vec, y_axis='mel',x_axis='s', fmax=fmax,sr = sr,ax=ax)
    #fig.tight_layout(pad=1.0)
    canvas.draw()
    (w,h) = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb() , dtype='uint8' )
    buf.shape = ( w, h,3 )
    w = int(w/scale)
    h = int(h/scale)
    img = Image.fromarray(buf).resize((h,w),Image.ANTIALIAS)
    buf =  np.asarray(img)
    return buf

