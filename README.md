# SPEECH EMOTION RECOGNISER

In Speech Emotion Recognition, the aim is to  recognize human emotion and affective states from speech. It capitalizes  the fact that voice often reflects underlying emotion through tone and pitch. This is also the phenomenon that animals like dogs and horses employ to be able to understand human emotion.

## OBJECTIVE

To build a model to recognize emotion from speech using the Librosa and NumPy libraries  and the RAVDESS dataset.

In this project, some libraries such as librosa, soundfile, and sklearn is used to build a model using the MLPClassifier.  It is a system through which various audio speech files can be classified into different emotions such as happy, sad, anger and neutral by computers.

The code works easily in python 3.8.x and there are compatibility issues in the python version 3.9

## DATASET

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song). Only the audio files are used.

## IMPLEMENTATION 

- Import necessary libraries by running `pip3 install -r requirements.txt` .
- If you are using the code for the first time run `ser.py` , to make the model and it will be stored in `result`.
- To know the emotion run `test.py` .

You can:
 
- Add more data to the folder `data`.
- edit the emotions specified can be edited in `utils.py` in `AVAILABLE_EMOTIONS` constant.

### Files included:

- requirements.txt - file that contains all the required dependencies for the project to run.
- test.py - The file that is to be run so that our voice is recorded and the emotion is identified
- ser.py - The actual speech emotion recogniser. It makes the model
- utils.py - The file that extracts features from the voice and classifies it into any one among angry, sad, neutral and happy
- convert_wavs.py - A utility script used for converting audio samples to be suitable for feature extraction

[Presentation Slides](https://docs.google.com/presentation/d/1ClvcQFMahFXRTfLWYMVtZs16s2Vgzat-s_iQCtyLyys/edit?usp=sharing)


## GROUP MEMBERS 

| NAME  | ROLL NUMBER |
| ------------- | ------------- |
| GAYATHRI P  | AM.EN.U4AIE20126  |
| LAKSHMI WARRIER  | AM.EN.U4AIE20143   |
| M DEVIKA  | AM.EN.U4AIE20144  |
| NIVEDITA RAJESH  | AM.EN.U4AIE20153 |
