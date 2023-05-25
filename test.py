#  Records voice

import pyaudio #to get audio from the user
import wave
import pickle #for saving the model
from sys import byteorder
from array import array
from struct import pack
import numpy as np

from utils import extract_feature

THRESHOLD = 500 #audio levels not normalised
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16 #signed 16-bit binary string
RATE = 16000

SILENCE = 30


def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD


def normalise(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r


def trim(snd_data):
    "Trim the silence at both the ends"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r


def record():
    """
    Record a word or words from the microphone and return the data as an array of signed shorts.

    Normalises the audio, trims silence from the start and end, and pads with 0.5 seconds of blank sound.

    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,input=True, output=True, frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        #this loop will run till sound is detected

        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
            #adding an extra second to know the seconds of silence 

        elif not silent and not snd_started:
            #sound started
            snd_started = True
            

        if snd_started and num_silent > SILENCE:
            #sound ends after 30 iterations without sound
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = trim(normalise(r))
    r = add_silence(r, 0.5)
    return sample_width, r


def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


if __name__ == "__main__":

    # load the saved model (after training)
    model = pickle.load(open("C:/Users/adith/Downloads/Speech-emotion-recogniser-main/Speech-emotion-recogniser-main/result/mlp_classifier.model", "rb"))
    print(" Please talk: ")
    filename = "test.wav"

    # record the file (start talking)
    record_to_file(filename)

    print(" Analysing the emotion.....\n")

    # extract features and reshape it 
    # As a result we get training arrays, which is used as classifiers to predict emotion
    features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1) 


    # predict
    result = model.predict(features)[0]

    print("\n")
    # show the result !
    if result == "happy":
        print(" Looks like it's a fine day for you. Cheers!")
        print(" Emotion: Happy 😄")

    elif result == "angry":
        print(" Calm down Hulk!")
        print(" Emotion: angry 😡")

    elif result == "sad":
        print(" Don't be sad. Remember, this too shall pass")
        print(" Emotion: sad ☹️")

    elif result == "neutral":
        print(" Emotion: neutral 😐")

    else:
        print(" Sorry! I couldn't read your emotion from the sound😔")
    print("\n")
