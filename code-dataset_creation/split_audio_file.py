import numpy as np
import os
from pydub import AudioSegment 
from pydub.utils import make_chunks


chunk_length = 1000
raw_audio_duration = '3hr'
category = 'music'

filenames_1hr = {'nature' : 'finalfinalnature.wav', 
				'music' : 'finalfinalmusic.wav', 
				'speech' : 'speech_sound.wav', 
				'machine' : 'machine_1hr.wav'}

filenames_3hr = {'nature' : 'final_nature_3hr.wav',
				 'music' : 'final_music_3hr.wav',
				 'machine' : 'final_machine_3hr.wav'}


src_path = os.path.join('raw_data','combined', raw_audio_duration, filenames_3hr[category])
dest_path = os.path.join('raw_data', 'segmented', raw_audio_duration, str(chunk_length)+'ms', category)

sr = 16000
myaudio = AudioSegment.from_file(src_path , "wav")
chunks = make_chunks(myaudio, chunk_length) #Make chunks of chunk_length msec
for i, chunk in enumerate(chunks):
	chunk_name = os.path.join(dest_path, "{}_{}_{}.wav".format(category, raw_audio_duration, i))
	print ("exporting", chunk_name)
	chunk.export(chunk_name, format="wav")

