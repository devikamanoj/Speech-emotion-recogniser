# How to Make a Speech Emotion Recognizer Using Python And Scikit-learn
To run this, you need to:
- `pip3 install -r requirements.txt`


Run `ser.py` to make the model. The model will be saved inside the directory `result`. This needs to be run only once as long as there is no change made in `ser.py` by you.
To test the model with our own voice, use `test.py`

You can:
- Tweak the model parameters ( or the whole model ) in `ser.py`.
- Add more data to `data` folder in condition that the audio samples are converted to 16000Hz sample rate and mono channel, `convert_wavs.py` does that.
- Editing the emotions specified in `utils.py` in `AVAILABLE_EMOTIONS` constant.