# add AWGN noise to ACF features

import numpy as np
#from scipy.io import wavfile
import librosa as lbr
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import warnings
import os

warnings.filterwarnings('ignore')
path = os.path.join('raw_data', '1hr')
filenames = {'nature' : 'finalfinalnature.wav', 
			'music' : 'finalfinalmusic.wav', 
			'speech' : 'speech_sound.wav', 
			'machine' : 'machine_1hr.wav'}

def extract_features(autocorr):
  zcpf = 0
  zidx = -1
  fmidx = -1
  fma = -1
  for idx in range(len(autocorr)-1):
    if zcpf==0 and autocorr[idx]*autocorr[idx+1] < 0:
      zcpf=1
      zidx=idx
      break
    #if autocorr[idx]>0 and zcpf==1 and idx!=0 and autocorr[idx+1]<autocorr[idx] and autocorr[idx-1]<autocorr[idx]:
    #   fmidx=idx;
    #   fma = autocorr[idx]
    # if zcpf==1 and fmidx!=-1:
    #   break
  if zcpf == 1:
    fmidx = zidx + np.argmax(autocorr[zidx+1:])
    #fma = autocorr[fmidx]

  return (zidx, fmidx, autocorr[fmidx])


audio_class = 'speech' # change to get features of different classes
file = filenames[audio_class]
frequency = 16000
noise_power_db = 0.1
noise_power_watt = 10 ** (noise_power_db / 10)
#frequency, signal = wavfile.read(os.path.join(path, file))

slice_length = 0.05 # frame length in seconds
overlap = 0.045 # overlap in seconds
slice_samples = int(slice_length * frequency)
overlap_samples = int(overlap * frequency)
#print(slice_samples, overlap_samples)
signal, sr =  lbr.load(os.path.join(path, file), sr=frequency)
slices = np.arange(0, len(signal), slice_samples-overlap_samples, dtype=np.int)
max_features = 755600
print(slices.shape[0])

mus_zcp = np.zeros((max_features+1))
mus_lag = np.zeros((max_features+1))
mus_peak = np.zeros((max_features+1))

scaler = MinMaxScaler(feature_range=(-1,1))

for i, start in enumerate(slices):
  end = start + slice_samples
  #print("sample no = {}, start = {}, end = {}".format(i, start, end))
  audio_slice = signal[start:end]
  if len(audio_slice) == 0:
  	break
  awgn_noise = np.random.normal(0, np.sqrt(noise_power_watt), audio_slice.shape)
  audio_slice = audio_slice + awgn_noise
  audio_slice_norm = audio_slice.reshape((audio_slice.shape[0], 1))
  scaler.fit(audio_slice_norm)
  audio_slice_norm = scaler.transform(audio_slice_norm)
  audio_slice_norm = audio_slice_norm.ravel()
  #autocorr = autocorrelation(audio_slice_norm)
  autocorr = acf(audio_slice_norm, nlags=slice_samples)
  zidx, fmidx, peak = extract_features(autocorr)
  print("{} -- zidx = {}, fmidx = {}, peak={}".format(i, zidx, fmidx, peak))
  mus_lag[i] = fmidx - zidx
  mus_zcp[i] = zidx
  mus_peak[i] = peak 
  if i == max_features:
    break


np.savetxt('ACF_features/noisy_weird_features-full/{}_peak.txt'.format(audio_class), mus_peak)
np.savetxt('ACF_features/noisy_weird_features-full/{}_zcp.txt'.format(audio_class), mus_zcp)
np.savetxt('ACF_features/noisy_weird_features-full/{}_lag.txt'.format(audio_class), mus_lag)