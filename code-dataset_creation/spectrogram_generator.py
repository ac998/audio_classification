import csv
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import matplotlib as mpl
from itertools import islice
import librosa
#               num_classes = len(self.classes)
#                print(len(self.classes))
""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.9, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)  

  
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

def plotstft(samples, samplerate, binsize=400, plotpath=None, colormap="jet"):
    plt.close('all')

    s = stft(samples, binsize)
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

#     timebins, freqbins = np.shape(ims)
#     print("timebins: ", timebins)
#     print("freqbins: ", freqbins)
    
#     w = ims.shape[0]/100.0
#     h = ims.shape[1]/100.0
    
    w = ims.shape[0]/1182.0
    
    h = ims.shape[1]/1182.0
    
    plt.figure(figsize=(w,h))
    plt.axis('off')
#     plt.axes([0.,0.,1.,1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    fig = plt.imshow(ims.T, origin="lower", aspect="auto", cmap=colormap, interpolation="nearest")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    
#     plt.colorbar()

#     plt.xlabel("time (s)")
#     plt.ylabel("frequency (hz)")
#     plt.xlim([0, timebins-1])
#     plt.ylim([0, freqbins])

#     xlocs = np.float32(np.linspace(0, timebins-1, 5))
#     plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
#     ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
#     yti = ylocs
#     print(freq[20])
#     print(freq[40])
#     print(freq[60])
#     plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

#     plotpath = 'temp.png'
    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight", pad_inches=0., dpi=1000)
#         image = cv2.imread(plotpath, 1)
#         print(image.shape)
    else:
        plt.show()

    plt.clf()

    return ims
fs = 16000
fsize = 8000
hop = 7200
hop_length = fsize // 40 
num_classes = 2
for k in range(1, num_classes):
    y, fs = librosa.load('Training/train_nature.wav', sr = None)
    rng = y.shape[0]//hop - 1
    print(rng,fs,y.shape[0])    
    for i in range(0, rng):
        data = y[i*hop:i*hop+fsize]
        path = './Training/NATURE_SPECTROGRAM/' + str(i+1) + '.png'
        ims = plotstft(data, fs, plotpath=path)
        
#print(data)

#path = r'C:\Users\pc\Downloads\TEST_CNN\cnn_test\file' + str(1) + '.png' 

    
