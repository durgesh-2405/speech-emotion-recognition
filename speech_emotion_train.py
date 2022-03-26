# importing libraries
import glob
import os
import sys

import librosa
import numpy as np
import soundfile
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
import warnings


# Function to extract features.
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result


# loading the data and extracting the features.
def load_data(test_size=0.2):
    file_list = glob.glob('/home/knoldus/Durgesh/TechHub/speech-emotion-recognition-ravdess-data/Actor_*/*.wav')
    # URL Link to the RAVDESS dataset -> {https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio}
    x, y = [], []

    for file in file_list:
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


# Main Function
if __name__ == "__main__":
    # Avoid printing warnings.
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # defining the emotion of RAVDESS dataset.
    emotions = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    # The emotion we will observe
    observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

    # Splitting the data set into train and test.
    x_train, x_test, y_train, y_test = load_data(test_size=0.25)

    # Printing the shape of train and test dataset
    # x_train.reshape()
    # x_test.reshape()
    print((x_train.shape[0], x_test.shape[0]))

    # The total number of features extracted.
    print(f'Features extracted: {x_train.shape[1]}')

    # Initializing the Multi Layer Perceptron Classifier to train.
    emotion_model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(700,),
                                  learning_rate='adaptive', max_iter=600)

    # Training the emotion_model.
    emotion_model.fit(x_train, y_train)
    print('Model Trained Successfully')

    # Making Predictions using test set
    y_pred = emotion_model.predict(x_test)

    # Calculating and Checking the accuracy of the model.
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Accuracy of the model is : {:.2f}%".format(accuracy * 100))

    # saving the trained model
    joblib.dump(emotion_model, 'speech_emotion_model.pkl')

    # loading the saved model
    saved_model = joblib.load('speech_emotion_model.pkl')

    # making prediction from saved model
    print(saved_model.predict(x_test))
