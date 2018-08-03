# # -*- coding: utf-8 -*-
# from chatterbot import ChatBot


# bot = ChatBot(
#     "Math & Time Bot",
#     logic_adapters=[
#         "chatterbot.logic.MathematicalEvaluation",
#         "chatterbot.logic.TimeLogicAdapter"
#     ],
#     input_adapter="chatterbot.input.VariableInputTypeAdapter",
#     output_adapter="chatterbot.output.OutputAdapter",
#     trainer='chatterbot.trainers.ChatterBotCorpusTrainer'
# )

# # Print an example of getting one math based response
# response = bot.get_response("What is 4 + 9?")
# print(response)

# # Print an example of getting one time based response
# response = bot.get_response("What time is it?")
# print(response)


import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import sys
import os
import pickle

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
	win = window(frameSize)
	hopSize = int(frameSize - np.floor(overlapFac * frameSize))
	samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
	cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
	samples = np.append(samples, np.zeros(frameSize))
	frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
	frames *= win
	return np.fft.rfft(frames)    

def logscale_spec(spec, sr=22000, factor=20.):
	timebins, freqbins = np.shape(spec)
	scale = np.linspace(0, 1, freqbins) ** factor
	scale *= (freqbins-1)/max(scale)
	scale = np.unique(np.round(scale))
	newspec = np.complex128(np.zeros([timebins, len(scale)]))
	for i in range(0, len(scale)):
		if i == len(scale)-1:
			newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
		else:
			newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)
	allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
	freqs = []
	for i in range(0, len(scale)):
		if i == len(scale)-1:
			freqs += [np.mean(allfreqs[int(scale[i]):])]
		else:
			freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]
	return newspec, freqs

def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
	samplerate, samples = wav.read(audiopath)
	s = stft(samples, binsize)
	sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
	ims = 20.*np.log10(np.abs(sshow)/10e-6)
	timebins, freqbins = np.shape(ims)
	freqbins=freqbins/2
	print("timebins: ", timebins)
	print("freqbins: ", freqbins)
	# plt.title('Spectrogram')
	# plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
	arr=[]
	fingerprint = []
	min_var=np.median(ims[0])
	for i in range(0,timebins,3):
		temp=np.median(ims[i])
		arr.append(temp)
		plt.plot(temp)
		if min_var > temp and temp>0:
			min_var = temp
		fingerprint.append(temp)
	if min_var<0:
		min_var = 0
	# plt.colorbar()
	# plt.xlabel("timebins ")
	# plt.ylabel("frequency (hz)")
	# plt.xlim([0, timebins-1])
	# plt.ylim([0, int(freqbins)])
	# plt.plot(arr,'.',color='b')
	# plt.show()
	# xlocs = np.float32(np.linspace(0, timebins-1, 5))
	# plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
	# ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
	# plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
	# if plotpath:
	# 	plt.savefig(plotpath, bbox_inches="tight")
	# plt.clf()
	return ims,arr,fingerprint

filename1='test.wav'
#ims2,arr2,fingerprint2=plotstft('newSong.wav')

def check_song(filename1,ims2,arr2,fingerprint2):
	ims,arr,fingerprint1 = plotstft(filename1)
	# ims2,arr2,fingerprint2 = plotstft(filename2)
	arrBig = fingerprint1
	arrSmall = fingerprint2
	l1 = len(fingerprint1)
	l2 = len(fingerprint2)
	err = 1000
	subsong = False
	sum1=0
	min_sum=20000
	newarr=[]
	for i in range(0,l1-l2+1):
		subArr = np.array(arrBig[i:i+l2])
		for j in range(0,l2):
			dummy = subArr[j]-arrSmall[j]
			if(dummy<0): dummy=dummy*(-1)
			newarr.append(dummy)
		newarr=np.array(newarr)
		sum1 = np.median(newarr)
		if sum1<=0:
			sum1 = sum1*(-1)
		if sum1<err:
			subsong=True
		newarr=[]
		if(min_sum>sum1):
			min_sum=sum1
	return subsong,min_sum

song_files = os.listdir('./songs')
main_lis={}

#############################

filename1='test.wav'
ims2,arr2,fingerprint1=plotstft(sys.argv[1])
fingerprint1=np.array(fingerprint1[20:])
filename2='db.pkl'

main_dir={}

def check_song1(fingerprint1):
	with open(filename2,'rb') as inp:
		main_lis = pickle.load(inp)
		for fprint in main_lis:
			arrBig = main_lis[fprint]
			arrSmall = fingerprint1
			l1 = len(arrBig)
			l2 = len(arrSmall)
			err = 1000
			subsong = False
			sum1=0
			min_sum=20000
			newarr=[]
			for i in range(0,l1-l2+1):
				subArr = np.array(arrBig[i:i+l2])
				for j in range(0,l2):
					dummy = subArr[j]-arrSmall[j]
					if(dummy<0): dummy=dummy*(-1)
					newarr.append(dummy)
				newarr=np.array(newarr)
				sum1 = np.median(newarr)
				if sum1<=0:
					sum1 = sum1*(-1)
				if sum1<err:
					subsong=True
				newarr=[]
				if(min_sum>sum1):
					min_sum=sum1
			main_dir[fprint]=min_sum	

check_song1(fingerprint1)
# print(main_dir)

main_dir = sorted(main_dir.items(),key = lambda x:x[1])
print(main_dir)
