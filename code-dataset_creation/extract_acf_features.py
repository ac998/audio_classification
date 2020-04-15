import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import warnings
import os

def extract_features(autocorr):
  zcpf = 0
  zidx = -1;
  fmidx = -1;
  fma = -1;
  for idx in range(len(autocorr)-1):
    if zcpf==0 and autocorr[idx]*autocorr[idx+1] < 0:
      zcpf=1
      zidx=idx
    if autocorr[idx]>0 and zcpf==1 and idx!=0 and autocorr[idx+1]<autocorr[idx] and autocorr[idx-1]<autocorr[idx]:
      fmidx=idx;
      fma = autocorr[idx]
    if zcpf==1 and fmidx!=-1:
      break

  return (zidx, fmidx, autocorr[fmidx])



warnings.filterwarnings('ignore')
path = os.path.join('raw_data_old', '1hr')
filenames = {'nature' : 'finalfinalnature.wav', 
			'music' : 'finalfinalmusic.wav', 
			'speech' : 'speech_sound.wav', 
			'machine' : 'machine_1hr.wav'}


audio_class = 'machine' # change to get features of different classes
file = filenames[audio_class]
frequency, signal = wavfile.read(os.path.join(path, file))
scaler = MinMaxScaler(feature_range=(-1,1))

slice_length = 0.05 # frame length in seconds
overlap = 0.045 # overlap in seconds
slice_samples = int(slice_length * frequency)
overlap_samples = int(overlap * frequency)
#print(slice_samples, overlap_samples)

slices = np.arange(0, len(signal), slice_samples-overlap_samples, dtype=np.int)
print(slices.shape)
max_features = 720510

mus_zcp = np.zeros((max_features+1))
mus_lag = np.zeros((max_features+1))
mus_peak = np.zeros((max_features+1))

for i, start in enumerate(slices):
  end = start + slice_samples
  #print("sample no = {}, start = {}, end = {}".format(i, start, end))
  audio_slice = signal[start:end]
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


np.savetxt('features/ACF/1hr/clean/{}_peak.txt'.format(audio_class), mus_peak)
np.savetxt('features/ACF/1hr/clean/{}_zcp.txt'.format(audio_class), mus_zcp)
np.savetxt('features/ACF/1hr/clean/{}_lag.txt'.format(audio_class), mus_lag)