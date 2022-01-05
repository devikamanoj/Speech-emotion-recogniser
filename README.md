# SPEECH EMOTION RECOGNISER

In Speech Emotion Recognition, the aim is to  recognize human emotion and affective states from speech. It capitalizes  the fact that voice often reflects underlying emotion through tone and pitch. This is also the phenomenon that animals like dogs and horses employ to be able to understand human emotion.

## OBJECTIVE

To build a model to recognize emotion from speech using the Librosa and NumPy libraries  and the RAVDESS dataset.

In this project, some libraries such as librosa, soundfile, and sklearn is used to build a model using the MLPClassifier.  It is a system through which various audio speech files can be classified into different emotions such as happy, sad, anger and neutral by computers.


## IMPLEMENTATION 

- Import necessary libraries by running `pip3 install -r requirements.txt` .
- If you are using the code for the first time run `ser.py` , to make the model and it will be stored in `result`.
- To know the emotion run `test.py` .
- `convert_wavs.py` converts all the audio samples to 16000Hz sample rate and mono channel

You can:
 
- Add more data to the folder `data`.
- The emotions specified can be edited in `utils.py` in `AVAILABLE_EMOTIONS` constant.

### Files included:

- requirements.txt - file that contains all the required dependencies for the project to run.
- convert_wavs.py - A utility script used for converting audio samples to be suitable for feature extraction
- ser.py - The actual speech emotion recogniser. It makes the model
- test.py - The file that is to be run so that our voice is recorded and the emotion is identified
- utils.py - The file that extracts features from the voice and classifies it into any one among angry, sad, neutral and happy
