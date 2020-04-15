import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QToolTip, QFileDialog,QButtonGroup,
                             QLabel, QRadioButton, QComboBox, QLineEdit, QPushButton, QGridLayout)
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QFont
from PyQt5 import QtCore
from playsound import playsound
from scipy import stats
import scipy.io
import numpy as np
import pandas as pd
import pygame
import time
from PyQt5 import QtGui
import python_speech_features
import librosa
import csv
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvus
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from sklearn.externals import joblib
import os
from statsmodels.tsa.stattools import acf
import warnings
import pickle
import sounddevice as sd
from scipy  import stats
from scipy.io.wavfile import write

warnings.filterwarnings('ignore')

class Window(QWidget): #inherits from the QWidget class. QWidget widget is the base class of all user interface objects in PyQt5.

        def __init__(self):
                super().__init__() #super() method returns the parent object of the Window class and we call its constructor. 
                self.initUI()
        def initUI(self):

                # #Naming The Widgets
                audioType = QLabel('Audio Input (.wav file): ')
                #audioType.setFont(QFont.setBold(self,True))
                fs = QLabel('Sampling Freq.(Hz)')
                time = QLabel('Duration(Sec)')
                predictionPart = QLabel('Prediction Part')
                myFont = QtGui.QFont()
                myFont.setBold(True)
                predictionPart.setFont(myFont)
                overlap = QLabel('50% Overlap')
                #-MFCC Feature extraction with 50% Overlap
                modelSelect = QLabel('Select The Model')
                frameSize = QLabel('Frame Size (in ms)')
                windowLenEdit = QLabel('Window Length(ms)')
                predicitonResult = QLabel('Total Prediction')
                self.lbl = QLabel(self)
                self.Index = 0
                self.modelName = 'SVM'
                self.featureName = 'ACF'
                self.FeatureModel = None
                self.Time = 0
                self.rate = 0
                self.cont_mc_cnt = 0
                self.cont_sp_cnt = 0
                self.cont_mu_cnt = 0
                self.cont_na_cnt = 0
                self.cont_mcna_cnt = 0
                self.cont_musp_cnt = 0
                pygame.init()
                self.final_y = []
                self.stopButtonBool = False
                
                #Implementing Those Widgets
                self.nameOfAudio = QLineEdit()
                self.uploadAudio = QPushButton("Upload Audio")
                self.fsEdit = QComboBox() 
                ##### Reording Button #####
                
                self.timeDuration = QLineEdit()
                self.stopButton = QPushButton("Stop Contionus Record")
                # self.stopButton.setEnabled(False)
                self.loadAudio = QPushButton("Load Data")
                self.recordAudio = QPushButton("Record Audio")
                self.loadAudio.setEnabled(False)
                self.plotEdit = QPushButton("Plot Data")
                self.CDRecord = QPushButton("Contionus Prediction Record")
                self.plotEdit.setEnabled(False)
                self.playFrame = QPushButton('Play Frame')
                self.playFrame.setEnabled(False)
                self.figure = plt.figure(figsize=(5,2),dpi=100)
                self.canvas = FigureCanvus(self.figure)
                self.toolbar = NavigationToolbar(self.canvas,self)
                self.processStart = QPushButton("Process")
                self.processStart.setEnabled(False)
                self.predictStart = QPushButton("Predict")
                self.predictStart.setEnabled(False)
                ##### Model Selection #####
                self.modelSelect = QComboBox()
                ##### Frame Size Selection #####
                self.frameSizeEdit = QComboBox()
                ##### Window Length Selection for Prediction #####
                self.windowLenEdit = QComboBox()
                self.modelGraph = QComboBox()
                self.Show = QPushButton("Show")
                self.Show.setEnabled(False)
                self.predictionRecord = QPushButton("Result")
                self.predictionRecord.setEnabled(False)
                self.totalPredictionResult = QLineEdit()
                self.cancelEdit = QPushButton("CANCEL")
                self.cancelEdit.setEnabled(False)
                self.back = QPushButton("<<")
                self.back.setEnabled(False)
                self.front = QPushButton(">>")
                self.front.setEnabled(False)
                self.showFrame = QLineEdit()
                self.startTime = QLineEdit()
                self.endTime = QLineEdit()
                self.reset = QPushButton('Reset')
                self.reset.setEnabled(False)
                
                
                #Filling In Details
                
                self.fsEdit.addItem('16000')
                self.fsEdit.addItem('44100')
                self.frameSizeEdit.addItem('50')
                self.frameSizeEdit.setEnabled(False)
                #self.frameSizeEdit.addItem('500')
                #self.modelSelect.addItem('SVM')
                self.modelSelect.addItem('ACF-SVM')
                self.modelSelect.addItem('MFCC-SVM')
                self.modelSelect.addItem('ACF-1DCNN')
                self.modelSelect.addItem('MFCC-1DCNN')
                #self.modelSelect.addItem('1D CNN Without MFCC')
                self.windowLenEdit.addItem('1000')
                self.windowLenEdit.addItem('500')
                self.windowLenEdit.addItem('750')
                self.windowLenEdit.setEnabled(False)
                #self.modelGraph.addItem('Model Plot')
                self.modelGraph.addItem('K-fold Accuracy')
                self.modelGraph.addItem('K-fold Loss')
                self.modelGraph.addItem('Confusion-Matrix')
                

                #Setting Layout
                grid = QGridLayout()
                grid.setSpacing(5)              
                #1st Row
                #grid.addWidget(audioType, 1, 0, 1, 1)
                grid.addWidget(self.nameOfAudio,1,1,1,2)
                grid.addWidget(self.uploadAudio,1,0,1,1)
                grid.addWidget(fs , 1, 3, 1, 1)
                grid.addWidget(self.fsEdit,1,4,1,1)
                grid.addWidget(self.CDRecord,1,5,1,1)
                
                #2nd Row
                grid.addWidget(self.recordAudio,2,0,1,1)
                grid.addWidget(self.loadAudio,2,1,1,1)
                grid.addWidget(self.plotEdit, 2, 2, 1, 1)
                grid.addWidget(time, 2, 3, 1, 1)
                grid.addWidget(self.timeDuration, 2, 4, 1, 1)
                grid.addWidget(self.stopButton,2,5,1,1)
                
                #3rd Row
                grid.addWidget(self.playFrame,3,0,1,1)
                grid.addWidget(self.toolbar, 3, 1, 1, 4)
                #4th Row
                grid.addWidget(self.canvas, 4, 0, 1, 4)
                grid.addWidget(self.lbl, 4,4,1,2)
                
                #5th Row
                grid.addWidget(predictionPart, 5, 2, 1, 1)
                
                #6th Row
                grid.addWidget(modelSelect, 6, 0, 1, 1)
                grid.addWidget(self.modelSelect, 6, 1, 1, 1)
                grid.addWidget(frameSize , 6, 2, 1, 1)
                grid.addWidget(self.frameSizeEdit, 6, 3, 1, 1)
                #grid.addWidget(self.modelGraph, 6, 4, 1, 1)
                grid.addWidget(self.reset, 6, 5, 1, 1)
                
                #7th Row
                grid.addWidget(windowLenEdit , 7, 0, 1, 1)
                grid.addWidget(self.windowLenEdit, 7, 1, 1, 1)
                grid.addWidget(self.processStart,7, 3, 1, 1)
                grid.addWidget(self.predictStart,7, 4, 1, 1)
                grid.addWidget(self.cancelEdit,7,5,1,1)

                #8th Row
                grid.addWidget(predicitonResult, 8, 0, 1, 1)
                grid.addWidget(self.totalPredictionResult, 8,1, 1, 3)
                self.totalPredictionResult.resize(220,80)
                grid.addWidget(self.predictionRecord, 8, 4, 1, 1)
                
                #9th Row
                grid.addWidget(self.back,9,0,1,1)
                grid.addWidget(self.startTime,9,1,1,1)
                grid.addWidget(self.showFrame,9,2,1,1)
                grid.addWidget(self.endTime,9,3,1,1)
                grid.addWidget(self.front,9,4,1,1)
                
                #10th row
                # grid.addWidget(self.reset,10,4,1,1)
                # grid.addWidget(self.cancelEdit,10,5,1,1)
                
                
                self.setLayout(grid)
                self.uploadAudio.clicked.connect(self.showFileUploadDialog)
                self.recordAudio.clicked.connect(self.recordLive)
                self.loadAudio.clicked.connect(self.load)
                self.plotEdit.clicked.connect(self.plot)
                self.playFrame.clicked.connect(self.playframe)
                self.modelSelect.currentIndexChanged.connect(self.identifyModelFeature)
                self.CDRecord.clicked.connect(self.continous_record)
                self.stopButton.clicked.connect(self.stopButtonClicked)
                
                self.processStart.clicked.connect(self.process)
                self.predictStart.clicked.connect(self.predict)
                self.predictionRecord.clicked.connect(self.record)
                self.Show.clicked.connect(self.modelShow)
                self.back.clicked.connect(self.left)
                self.front.clicked.connect(self.right)
                self.reset.clicked.connect(self.Reset)
                self.cancelEdit.clicked.connect(self.cancel)
                
                                
                self.setGeometry(300, 300, 500, 400) #locates the window on the screen and sets it size(x,y,x+w,y+d)
                self.setWindowTitle('GUI for Audio Scene Prediction')
                #self.show(QIcon('FileName.png'))
                self.show()


        def extractFeatures(self,autocorr):
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
        
        def str2int_fs(self):
            b = [int(y) for y in self.fsEdit.currentText()]
            c = 0;
            for i in b:
                c = c*10 + i
            return c

        def stopButtonClicked(self):
            print("HELL")
            self.stopButtonBool = True
            print("Total Time Recorded: ",time.perf_counter()-self.start_time_rec)
            print("Total Time stored: ", i)
            self.record()

        def str2int_framesize(self):
            b = [int(y) for y in self.frameSizeEdit.currentText()]
            c = 0;
            for i in b:
                c = c*10 + i
            return c
        
        def str2int_winlen(self):
            b = [int(y) for y in self.windowLenEdit.currentText()]
            c = 0;
            for i in b:
                c = c*10 + i
            return c
        
        def showFileUploadDialog(self):
                self.fname = QFileDialog.getOpenFileName(self,
                                                         'Open Recorded Audio',
                                                         'test_audio/',
                                                         'Audio files (*.wav *.mp3)')
                self.nameOfAudio.setText(self.fname[0])
                self.loadAudio.setEnabled(True)
                
        def load(self):
            fs1 = self.str2int_fs()
            (self.wavFileR,self.rate) = librosa.load(self.nameOfAudio.displayText(),sr=int(fs1),mono=True)
            self.wavFile = self.cleanse(self.wavFileR)
            time_duration = self.wavFile.size/self.rate
            pr = str(round(time_duration,2)) + " Sec"
            self.timeDuration.setText(pr)
            self.plotEdit.setEnabled(True)
            self.processStart.setEnabled(True)
            self.Show.setEnabled(True)
            self.reset.setEnabled(True)
            self.cancelEdit.setEnabled(True)

        def keyPressEvent(self, e):
            if e.key() == Qt.Key_Escape:
                print("HELL")
                self.close()
            

        def cleanse(self,wavFile):
            #declare amn empty wav file  as self.WavFile
            new_wavfile = []
            block_slice_length = 2 # frame length in seconds
            block_slice_samples = int(block_slice_length * self.rate)
            
            block_slices = np.arange(0, len(wavFile), block_slice_samples, dtype=np.int)
            print(block_slices.shape)

            for i, start in enumerate(block_slices):
                end = start + block_slice_samples
               #print("sample no = {}, start = {}, end = {}".format(i, start, end))
                block_slice = wavFile[start:end]
                block_max = np.max(np.abs(block_slice)) 
                max_threshold =0.55*block_max
                min_threshold = -0.55*block_max
                frame_slice_length = 0.05
                frame_slice_samples = int(frame_slice_length*self.rate)
                frame_slices = np.arange(0,len(block_slice),frame_slice_samples,dtype=np.int)
                for j, begin in enumerate (frame_slices):
                    last = begin + frame_slice_samples
                    frame_slice = block_slice[begin:last]
                    frame_slice_list = list(frame_slice)
                    frame_slice_intermediate = list(filter(lambda x: x>=min_threshold and x<=max_threshold, frame_slice_list))
                    new_wavfile += frame_slice_intermediate
                    # clean_frame = np.array(frame_slice_intermediate)
                    # new_wavfile.concat()
            new_wavfile = np.array(new_wavfile)
            return new_wavfile
       
        def Identify(self,index):
            if self.featureName == 'MFCC':
               a = ['Machine', 'Music', 'Speech', 'Nature']
            elif self.featureName == 'ACF':
               a = ['Music and Speech', 'Machine and Nature'] #ulta hai
            print(index)
            return (a[int(self.y[int(index)])])
            
        def left(self):
            self.front.setEnabled(True)
            self.Index -= 1
            if(self.Index<=0):
                self.back.setEnabled(False)
            
            self.frameplot()
            start = "<< "+"{:.3f}".format(self.Index*self.Time)+' sec.'
            self.startTime.setText(start)
            end = "{:.3f}".format((self.Index+1)*self.Time)+' sec. >>'
            self.endTime.setText(end)
            show = self.Identify(self.Index)
            p = "Frame " + str(self.Index+1) + " || " + show
            self.showFrame.setText(p)
                
        def right(self):
            self.back.setEnabled(True)
            self.Index += 1
            if (self.Index>=self.size-1):
                self.front.setEnabled(False)
            
            self.frameplot()
            start = "<< "+"{:.3f}".format(self.Index*self.Time)+' sec.'
            self.startTime.setText(start)
            end = "{:.3f}".format((self.Index+1)*self.Time)+' sec. >>'
            self.endTime.setText(end)
            show = self.Identify(self.Index)
            p = "Frame " + str(self.Index+1) + " || " + show
            self.showFrame.setText(p)
            
        def plot(self):
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            x = np.arange(1,self.wavFile.size+1)
            x = np.divide(x,self.rate)
            ax.plot(x,self.wavFile,'b-')
            ax.set_title('Uploaded Audio')
            self.canvas.draw()
            self.playFrame.setEnabled(True)
            self.passWavFile = self.wavFile
               
        def frameplot(self):   
            self.playFrame.setEnabled(True)
            self.figure.clear()
            start = int((float(self.Index)* float(str(self.Time)) *float(self.rate)))
            end = int((float(self.Index+1)* float(self.Time) *float(self.rate))-1)
            wave = self.wavFile[start:end]
            x = np.arange(1,wave.size+1)
            x = np.divide(x,self.rate)
            x = np.add(x,self.Index*self.Time)
            ax = self.figure.add_subplot(111)
            ax.plot(x,wave,'b-')
            ax.set_title('Frame Number '+str(self.Index+1))
            self.canvas.draw()
            self.passWavFile = wave
            
        def playframe(self):
            sd.play(self.passWavFile,self.rate)

        def identifyModelFeature(self):
        	self.FeatureModel = self.modelSelect.currentText().split("-")
        	self.featureName = self.FeatureModel[0]
        	self.modelName = self.FeatureModel[1]
        	if self.featureName == 'MFCC':
        		self.windowLenEdit.setEnabled(True)
        	if self.featureName == 'ACF':
        		self.windowLenEdit.setCurrentIndex(0)
        		self.windowLenEdit.setEnabled(False)
        	print(self.featureName)
        	print(self.modelName)
            
        def process(self):
            if self.featureName == 'MFCC':
        	    self.frameSize = self.str2int_winlen()
            else:
                self.frameSize = self.str2int_framesize()
            # print('hell')
            print(self.frameSize)
            self.overLap = self.frameSize
            print(self.overLap)
            print(self.rate*self.frameSize/1000)
            if self.featureName == 'MFCC':
                # print(math.log2(self.rate*self.frameSize/1000))
                # print(math.ceil(math.log2(self.rate*self.frameSize/1000)))
                # self.nfft = 2**(math.ceil(math.log2(self.rate*self.frameSize/1000)))
                # self.mfcc = python_speech_features.base.mfcc(self.wavFile, samplerate=self.rate, winlen=self.frameSize/1000, winstep=self.overLap/1000, numcep=26, nfilt=26,
                #         nfft=self.nfft, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True)
                # self.csvData = self.mfcc
                
                n_samples = (self.frameSize//1000) * self.rate
                print(len(self.wavFile))
                print(n_samples)
                slices = np.arange(0, len(self.wavFile), n_samples, dtype=np.int)
                print(slices)
                mfcc_arr = np.zeros((slices.shaep[0], 26))
                print(mfcc_arr.shape)
                start = 0
                for i, start in enumerate(slices):
                    wave_l = self.wavFile[start:start+n_samples]
                    mfcc = np.mean(librosa.feature.mfcc(y=wave_l, sr=self.rate, n_mfcc=26), axis=1)
                    mfcc_arr[i, : ] = mfcc 
                self.csvData = mfcc_arr
                np.savetxt('predictionMFCC.csv', self.csvData)
            elif self.featureName=='ACF': 
                scaler = MinMaxScaler(feature_range=(-1,1))
                slice_length = 0.05 # frame length in seconds
                overlap = 0.045 # overlap in seconds
                slice_samples = int(slice_length * self.rate)
                overlap_samples = int(overlap * self.rate)
                
                slices = np.arange(0, len(self.wavFile), slice_samples-overlap_samples, dtype=np.int)
                print(slices.shape)
                max_features = slices.shape[0]-1

                arr_zcp = np.zeros((max_features+1))
                arr_lag = np.zeros((max_features+1))
                arr_peak = np.zeros((max_features+1))

                for i, start in enumerate(slices):
                    end = start + slice_samples
                   #print("sample no = {}, start = {}, end = {}".format(i, start, end))
                    audio_slice = self.wavFile[start:end]
                    audio_slice_norm = audio_slice.reshape((audio_slice.shape[0], 1))
                    scaler.fit(audio_slice_norm)
                    audio_slice_norm = scaler.transform(audio_slice_norm)
                    audio_slice_norm = audio_slice_norm.ravel()
                   #autocorr = autocorrelation(audio_slice_norm)
                    if (len(audio_slice_norm) == 0):
                        break
                    autocorr = acf(audio_slice_norm, nlags=slice_samples)
                    zidx, fmidx, peak = self.extractFeatures(autocorr)
                   #print("{} -- zidx = {}, fmidx = {}, peak={}".format(i, zidx, fmidx, peak))
                    arr_lag[i] = fmidx - zidx
                    arr_zcp[i] = zidx
                    arr_peak[i] = peak 
                    if i == max_features:
                        break
                self.csvData = np.column_stack((arr_lag, arr_peak, arr_zcp))
                np.savetxt('predictionACF.csv', self.csvData)
            self.predictStart.setEnabled(True)	
                
       
        def predict(self):
            if self.featureName == 'ACF':
                if self.modelName == '1DCNN':
                    self.scaler = joblib.load('../models/ACF/1D-CNN/scaler_1d_cnn_v2.sav')
                    self.my_model = load_model('../models/ACF/1D-CNN/1d_cnn_v2.hdf5')
                elif self.modelName == 'SVM': 
                    #self.my_model = pickle.load(open('../models/ACF/SVM/weight_without_gridsearch_v1','rb'))
                    self.scaler = joblib.load('../models/ACF/SVM/scaler_without_gridsearch_v1.sav')
                    self.my_model = joblib.load('../models/ACF/SVM/weight_without_gridsearch_v1.sav')
                feat = np.loadtxt('predictionACF.csv')
            else:
                if (self.modelName=='SVM'):
                    if self.windowLenEdit.currentText() == '500':
                        self.scaler = joblib.load('../models/MFCC/SVM/500/scaler_without_gridsearch_v1.sav')
                        self.my_model = joblib.load('../models/MFCC/SVM/500/weight_without_gridsearch_v1.sav')
                    elif self.windowLenEdit.currentText() == '750':
                        self.scaler = joblib.load('../models/MFCC/SVM/750/scaler_without_gridsearch_v1.sav')
                        self.my_model = joblib.load('../models/MFCC/SVM/750/weight_without_gridsearch_v1.sav')
                    elif self.windowLenEdit.currentText() == '1000':
                        self.scaler = joblib.load('../models/MFCC/SVM/1000/scaler_without_gridsearch_v1.sav')
                        self.my_model = joblib.load('../models/MFCC/SVM/1000/weight_without_gridsearch_v1.sav')
                elif (self.modelName=='1DCNN'):
                    if self.windowLenEdit.currentText() == '500':
                        self.scaler = joblib.load('../models/MFCC/1D-CNN/500/scaler_1d_cnn_v1.sav')
                        self.my_model = load_model('../models/MFCC/1D-CNN/500/1d_cnn_v1.hdf5')
                    elif self.windowLenEdit.currentText() == '750':
                        self.scaler = joblib.load('../models/MFCC/1D-CNN/750/scaler_1d_cnn_v1.sav')
                        self.my_model = load_model('../models/MFCC/1D-CNN/750/1d_cnn_v1.hdf5')
                    elif self.windowLenEdit.currentText() == '1000':
                        self.scaler = joblib.load('../models/MFCC/1D-CNN/1000/scaler_1d_cnn_v1.sav')
                        self.my_model = load_model('../models/MFCC/1D-CNN/1000/1d_cnn_v1.hdf5')

                feat = np.loadtxt('predictionMFCC.csv')
            print(len(feat))
            feat = self.scaler.transform(feat)
            feat = np.nan_to_num(feat)
            if self.modelName == 'SVM':
                self.y = self.my_model.predict(feat)
            elif self.modelName =='1DCNN':
                feat = feat.reshape((feat.shape[0], feat.shape[1], 1))
                self.y = self.my_model.predict(feat)
                self.y = np.argmax(self.y, axis=1)
                print("type of prediction ", type(self.y))
                print("shape of prediction ", self.y.shape)

            #self.y = self.y.ravel()
            self.final_y += list(self.y)

            self.predictionRecord.setEnabled(True)
            
            print(type(self.Time))
            self.Time = self.frameSize/1000
            print(type(self.Time))
            if self.featureName=='ACF':
                for res in self.y:
                    if res == 1: 
                        #print("machine+Nature")
                        self.cont_mcna_cnt = self.cont_mcna_cnt+1
                    elif res == 0: 
                        #print("music+Speech")
                        self.cont_musp_cnt = self.cont_musp_cnt+1
                    print("Mahince and Nature Count: ", self.cont_mcna_cnt, " Music and Speech Count: ", self.cont_musp_cnt)
                    pass
                os.remove('predictionACF.csv')
            elif self.featureName == 'MFCC':
                for res in self.y:
                    if res == 0: 
                        #print("machine")
                        self.cont_mc_cnt = self.cont_mc_cnt+1
                    elif res == 1: 
                        #print("music")
                        self.cont_mu_cnt = self.cont_mu_cnt+1
                    elif res == 2: 
                        #print("speech")
                        self.cont_sp_cnt = self.cont_sp_cnt+1
                    elif res == 3: 
                        #print("nature")
                        self.cont_na_cnt = self.cont_na_cnt+1
                    print("Mahince_Count: ", self.cont_mc_cnt, " Music Count: ", self.cont_mu_cnt, " Speech Count: ", self.cont_sp_cnt, " Nature count: ", self.cont_na_cnt)
                os.remove('predictionMFCC.csv')

                    
        def record(self):
            self.Index = 0
            self.front.setEnabled(True)
            self.final_y = np.array(self.final_y)
            self.size = self.final_y.shape[0]
            if self.featureName == 'MFCC':
                c_machine = sum(self.final_y==0)
                c_music = sum(self.final_y==1)
                c_speech = sum(self.final_y==2)
                c_nature = sum(self.final_y==3)
                pr = 'Ma: '+"{:.2f}".format(100*c_machine/self.size) + '||' + 'Mu: ' + "{:.2f}".format(100*c_music/self.size) + '||' + 'N: '+"{:.2f}".format(100*c_nature/self.size)+ '||' + 'S: '+"{:.2f}".format(100*c_speech/self.size)
                

            elif self.featureName == 'ACF':
                c_music_speech = sum(self.final_y==0)
                c_machine_nature = sum(self.final_y==1)
                pr = 'MS: '+"{:.2f}".format(100*c_music_speech/self.size) + '||' + 'MN: ' + "{:.2f}".format(100*c_machine_nature/self.size)                
            self.totalPredictionResult.setText(pr)

            
            self.frameplot()
            show = self.Identify(self.Index)
            p = "Frame "+str(self.Index+1) + " || " + show
            self.startTime.setText('<< 0 sec')
            self.showFrame.setText(p)
            self.endTime.setText(str(self.Time)+' sec >>')
            
        def modelShow(self):
            img_name = self.modelGraph.currentText()
            frameS = self.frameSizeEdit.currentText()
            modelN = self.modelSelect.currentText()
            image_name = frameS+'_'
            if(modelN=='FNN'):
                image_name += 'FNN_'
            elif(modelN=='1D CNN'):
                image_name += 'CNN_'
            if(img_name=='K-fold Accuracy'):
                image_name += 'acc'
            elif(img_name=='K-fold Loss'):
                image_name += 'loss'
            else:
                image_name += 'cm'
        
            pixmap = 	(image_name+'.png')
            self.lbl.setPixmap(pixmap)

        def recordLive(self):
            #save.mp3
            fs = 44100  # Sample rate
            seconds = 5  # Duration of recording
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
            sd.wait()  # Wait until recording is finished
            write('output.wav', fs, myrecording)  # Save as WAV file 
            self.nameOfAudio.setText('output.wav')
            self.loadAudio.setEnabled(True)

        def continous_record(self):
            if self.stopButton.isEnabled():
                print("Success1")
            self.stopButton.setEnabled(True)
            if self.stopButton.isEnabled():
                print("Success2")
            # self.start_time_rec = time.perf_counter()
            # self.threadClass = ThreadClass()
            # self.threadClass.start()
            while(1):
                fs = 44100  # Sample rate
                seconds = 3  # Duration of recording
                myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
                sd.wait()  # Wait until recording is finished
                write('output.wav', fs, myrecording)  # Save as WAV file 
                self.nameOfAudio.setText('output.wav')
                (self.wavFileR,self.rate) = librosa.load(self.nameOfAudio.displayText(),sr=int(fs),mono=True)
                self.wavFile = self.cleanse(self.wavFileR)
                self.process()
                self.predict()
            
        def cancel(self):
                print('Cancelled')
                self.close()
        
        def Reset(self):
            self.figure.clear()
            self.loadAudio.setEnabled(False)
            self.plotEdit.setEnabled(False)
            self.playFrame.setEnabled(False)
            self.processStart.setEnabled(False)
            self.Show.setEnabled(False)
            self.back.setEnabled(False)
            self.front.setEnabled(False)
            self.predictionRecord.setEnabled(False)
            self.predictStart.setEnabled(False)
            self.Index = 0
            self.figure.clear()
            self.nameOfAudio.setText(' ')
            self.timeDuration.setText('')
            self.totalPredictionResult.setText('')
            self.showFrame.setText('')
            self.startTime.setText('')
            self.endTime.setText('')
            self.lbl.clear()
            self.reset.setEnabled(False)
            self.frameSizeEdit.setEnabled(False)
            self.windowLenEdit.setEnabled(False)

class ThreadClass(QtCore.QThread):
    def __init_(self,parent=None):
        super(ThreadClass,self).__init__(parent)

    def run(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.handleNewData)
        timer.start(1000)
        # keep reference to timer        
        self.timer = timer
        mic = MicrophoneRecorder()
        mic.start()  

        # keeps reference to mic        
        self.mic = mic

    def handleNewData(self):
        """ handles the asynchroneously collected sound chunks """        
        # gets the latest frames        
        frames = self.mic.get_frames()
        
        if len(frames) > 0:
            # keeps only the last frame
            current_frame = frames[-1]
            # # plots the time signal
            # self.line_top.set_data(self.time_vect, current_frame)
            # # computes and plots the fft signal            
            # fft_frame = np.fft.rfft(current_frame)
            # if self.autoGainCheckBox.checkState() == QtCore.Qt.Checked:
            #     fft_frame /= np.abs(fft_frame).max()
            # else:
            #     fft_frame *= (1 + self.fixedGainSlider.value()) / 5000000.
            #     #print(np.abs(fft_frame).max())
            # self.line_bottom.set_data(self.freq_vect, np.abs(fft_frame))            
            
            # # refreshes the plots
            # self.main_figure.canvas.draw()


if __name__ == '__main__':
##
    app = QApplication(sys.argv) #Every PyQt5class Window(QWidget): #inherits from the QWidget class. QWidget widget is the base class of all user interface objects in PyQt5.
    window = Window()
    sys.exit(app.exec_()) #enters the mainloop of the application. The event handling starts from this point.
