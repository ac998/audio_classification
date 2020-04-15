import numpy as np
import os 

raw_audio_duration_per_category = '1hr'
chunk_length = 1000
categories = ['speech', 'music', 'machine', 'nature']
dest_path = os.path.join('sets_labels', 'MFCC', raw_audio_duration_per_category, str(chunk_length)+'ms')
train_test_split = 0.8

# loading machine features
machine_file = os.path.join('features', 'MFCC', raw_audio_duration_per_category, str(chunk_length)+'ms', 
							'{}_machine_mfcc_{}.csv'.format(raw_audio_duration_per_category, str(chunk_length)+'ms'))
print(machine_file)
mach_mfcc = np.loadtxt(machine_file)
print("machine feature shape : ", mach_mfcc.shape)

# loading speech features
speech_file = os.path.join('features', 'MFCC', raw_audio_duration_per_category, str(chunk_length)+'ms', 
							'{}_speech_mfcc_{}.csv'.format(raw_audio_duration_per_category, str(chunk_length)+'ms'))
print(speech_file)
spe_mfcc = np.loadtxt(speech_file)
print("speech feature shape : ", spe_mfcc.shape)

# loading nature features
nature_file = os.path.join('features', 'MFCC', raw_audio_duration_per_category, str(chunk_length)+'ms', 
							'{}_nature_mfcc_{}.csv'.format(raw_audio_duration_per_category, str(chunk_length)+'ms'))
print(nature_file)
nat_mfcc = np.loadtxt(nature_file)
print("nature feature shape : ", nat_mfcc.shape)

# loading music features
music_file = os.path.join('features', 'MFCC', raw_audio_duration_per_category, str(chunk_length)+'ms', 
							'{}_music_mfcc_{}.csv'.format(raw_audio_duration_per_category, str(chunk_length)+'ms'))
print(music_file)
mus_mfcc = np.loadtxt(music_file)
print("speech feature shape : ", mus_mfcc.shape)

# creating labels
# machine - 0
# music - 1
# speech - 2
# nature - 3
mach_labels = np.zeros((mach_mfcc.shape[0], 1), dtype=np.int8)
mus_labels = np.ones((mus_mfcc.shape[0], 1), dtype=np.int8)
spe_labels = 2 * np.ones((spe_mfcc.shape[0], 1), dtype=np.int8)
nat_labels = 3 * np.ones((nat_mfcc.shape[0], 1), dtype=np.int8)

# concatenating sets
feats = np.vstack((mach_mfcc, mus_mfcc, spe_mfcc, nat_mfcc))
labels = np.vstack((mach_labels, mus_labels, spe_labels, nat_labels))

# shuffling sets
rand_idx = np.random.choice(np.arange(feats.shape[0]), size=feats.shape[0], replace=False)
feats = feats[rand_idx, : ]
labels = labels[rand_idx, : ] 
print ("dataset shape : ", feats.shape)
print ("labels shape : ", labels.shape)

# splitting into training and testing 
split_idx = int(train_test_split * feats.shape[0])
train_set = feats[:split_idx, :]
test_set = feats[split_idx:, :]
train_labels = labels[:split_idx, :]
test_labels = labels[split_idx:, :]
print("train set shape : ", train_set.shape)
print("train labels shape : ", train_labels.shape)
print("test set shape : ", test_set.shape)
print("test labels shape : ", test_labels.shape)


np.savetxt(os.path.join(dest_path, 'train_set_{}_{}ms.csv'.format(raw_audio_duration_per_category, str(chunk_length))), train_set)
np.savetxt(os.path.join(dest_path, 'train_labels_{}_{}ms.csv'.format(raw_audio_duration_per_category, str(chunk_length))), train_labels)
np.savetxt(os.path.join(dest_path, 'test_set_{}_{}ms.csv'.format(raw_audio_duration_per_category, str(chunk_length))), test_set)
np.savetxt(os.path.join(dest_path, 'test_labels_{}_{}ms.csv'.format(raw_audio_duration_per_category, str(chunk_length))), test_labels)