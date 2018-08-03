# import numpy as np 

# arrBig = np.random.rand(10)
# arrSmall = np.random.rand(5)

# l1 = len(arrBig)
# l2 = len(arrSmall)

# err = 5;
# subsong = False

# for i in range(0,l1-l2+1):
# 	subArr = np.array(arrBig[i:i+l2])
# 	sum1 = np.sum(subArr - arrSmall)
# 	if sum1<=0:
# 		sum1 = sum1*(-1)
# 	if sum1<err:
# 		subsong=True
# print(arrBig,arrSmall)
# print(subsong)


from pydub import AudioSegment
t1=1
t2=100
t1 = t1 * 1000 #Works in milliseconds
t2 = t2 * 1000
t3 = 200000
newAudio = AudioSegment.from_wav("../songs/eigenvalue.wav")
newAudio1 = newAudio[t1:t2]
newAudio1.export('../songs/eigenvalue_split1.wav', format="wav")

newAudio2 = newAudio[t2:t3]
newAudio2.export('../songs/eigenvalue_split2.wav', format="wav")

newAudio3 = newAudio[t3:]
newAudio3.export('../songs/eigenvalue_split3.wav', format="wav")
