# to extract deatures from the recorded sound

import soundfile #read, write and open sound file
import numpy as np #for ndarray in sound file
import librosa # help in feature extraction
import glob #to load data from the library
import os # to get the basename of the library
from sklearn.model_selection import train_test_split #for training, testing and splitting data

# all emotions on RAVDESS dataset
int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# we allow only these emotions
AVAILABLE_EMOTIONS = {
    "angry",
    "sad",
    "neutral",
    "happy"
}


def extract_feature(file_name, **kwargs):
    """
    Extract features (mfcc, chroma, mel, contrast, tonnetz) from audio file `file_name`
    """
    mfcc = kwargs.get("mfcc") #Mel Frequency Cepstral Coefficients
    chroma = kwargs.get("chroma") #Representation for audio where spectrum is projected onto 12 bins representing the 12 distinct semitones 
    mel = kwargs.get("mel") # deals with human perception of frequency and sclae of pitches
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    #to open sound from dataset, we use soundfile module
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc: # representation of short-time power spectrum of sound
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz( y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
#return the sound features of requires soundfile in arrays
    return result


def load_data(test_size=0.2):
    X, y = [], []
    # for loading data from the folder
    for file in glob.glob("data/Actor_*/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)

        # get the emotion label
        emotion = int2emotion[basename.split("-")[2]]

        # this model detects only the 4 emotions mentioned
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extract speech features
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # add to data
        X.append(features)
        y.append(emotion)
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)
