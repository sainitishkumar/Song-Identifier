import pydub
import sys
sound = pydub.AudioSegment.from_mp3("../songs/1.mp3")
sound.export('/Users/sainitish/Desktop/1.wav',format="wav")

sound = pydub.AudioSegment.from_mp3('../songs/The-Lights-Galaxia--While-She-Sleeps.mp3')
sound.export('../songs/The-Lights-Galaxia--While-She-Sleeps.wav',format='wav')