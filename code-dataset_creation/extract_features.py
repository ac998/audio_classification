import numpy as np
import librosa as lbr
import os

n_mfcc = 26
chunk_length = 1000
raw_audio_duration = '1hr'
sr = 16000
category = 'nature'

src_path = os.path.join('raw_data', 'segmented', raw_audio_duration,  str(chunk_length)+'ms', category)
dest_path = os.path.join('features', 'MFCC', raw_audio_duration, str(chunk_length)+'ms', 
						'{}_{}_mfcc_{}.csv'.format(raw_audio_duration, category, str(chunk_length)+'ms'))
print(src_path)
print(dest_path)

files = os.listdir(src_path)
features = np.zeros((len(files), n_mfcc))
for i, fname in enumerate(files):
    wave,sr = lbr.load(os.path.join(src_path, fname), sr=sr)
    mfcc = np.mean(lbr.feature.mfcc(y=wave, sr=sr, n_mfcc=n_mfcc), axis=1)
    features[i, : ] = mfcc
    print(sr, i)

np.savetxt(dest_path, features)

