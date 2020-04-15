import numpy as np
import os
import matplotlib.pyplot as plt

lag, peak, zcp = 'lag', 'peak', 'zcp'
parent = 'ACF_features/noisy_features'
features_to_plot = 1000


features = [lag, peak, zcp]

for feature_name in features:
	mus = np.loadtxt(os.path.join(parent, 'music_{}.txt'.format(feature_name)))[:features_to_plot]
	nat = np.loadtxt(os.path.join(parent, 'nature_{}.txt'.format(feature_name)))[:features_to_plot]
	spe = np.loadtxt(os.path.join(parent, 'speech_{}.txt'.format(feature_name)))[:features_to_plot]
	mac = np.loadtxt(os.path.join(parent, 'machine_{}.txt'.format(feature_name)))[:features_to_plot]
	sample_no = mus.size
	fig = plt.figure(feature_name, figsize=(15, 10))
	plt.scatter(mus, np.arange(sample_no), color='black', marker='*', label='music')
	plt.scatter(nat, np.arange(sample_no), color='green', marker='D', label='nature')
	plt.scatter(spe, np.arange(sample_no), color='red', marker='x', label='speech')
	plt.scatter(mac, np.arange(sample_no), label='machine')
	if feature_name == peak:
		plt.xlim([0, 1])
	else:
		plt.xlim([0, 100])
	plt.legend()
	plt.title(feature_name)
	plt.show()
	#plt.savefig('{}.png'.format(feature_name)) #comment plt.show() and uncomment this to save
