import multiprocessing
import pickle
import threading
from os import listdir
from os.path import isfile, join
from queue import Queue
from timeit import default_timer as timer

import keras
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

load = True
save = False


class Diagnosis:
    def __init__(self, id, diagnosis, image_path):
        self.id = id
        self.diagnosis = diagnosis
        self.image_path = image_path


def get_wav_files():
    audio_path = 'audio_and_txt_files/'
    files = [f for f in listdir(audio_path) if isfile(join(audio_path, f))]  # Gets all files in dir
    wav_files = [f for f in files if f.endswith('.wav')]  # Gets wav files
    wav_files = sorted(wav_files)
    return wav_files, audio_path


def diagnosis_data():
    diagnosis = pd.read_csv('patient_diagnosis.csv')
    wav_files, audio_path = get_wav_files()
    diag_dict = {101: "URTI"}
    diagnosis_list = []

    for index, row in diagnosis.iterrows():
        diag_dict[row[0]] = row[1]

    c = 0
    for f in wav_files:
        diagnosis_list.append(Diagnosis(c, diag_dict[int(f[:3])], audio_path + f))
        c += 1

    return diagnosis_list


def audio_features(filename):
    sound, sample_rate = librosa.load(filename, res_type='kaiser_fast')

    stft = np.abs(librosa.stft(sound))
    mfccs = np.median(librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=40), axis=1)
    chroma = np.median(librosa.feature.chroma_stft(S=stft, sr=sample_rate), axis=1)
    mel = np.median(librosa.feature.melspectrogram(sound, sr=sample_rate), axis=1)
    contrast = np.median(librosa.feature.spectral_contrast(S=stft, sr=sample_rate), axis=1)
    tonnetz = np.median(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sample_rate), axis=1)
    rms = np.median(librosa.feature.rms(S=stft), axis=1)
    poly = np.median(librosa.feature.poly_features(S=stft, sr=sample_rate, n_fft=100), axis=1)

    concat = np.concatenate((tonnetz, rms, poly, contrast, mel, chroma, mfccs))
    return concat


queue = Queue()
storage = []


def target():
    to_hot_one = {"COPD": 0, "Healthy": 1, "URTI": 2, "Bronchiectasis": 3, "Pneumonia": 4, "Bronchiolitis": 5,
                  "Asthma": 6, "LRTI": 7}
    while not queue.empty():
        f = queue.get()
        diag = to_hot_one[f.diagnosis]
        attrs = audio_features(f.image_path)
        storage.append((diag, attrs))
        queue.task_done()


def data_points():
    for f in diagnosis_data():
        queue.put(f)

    for x in range(multiprocessing.cpu_count()):
        task = threading.Thread(target=target)
        task.daemon = True
        task.start()

    start = timer()
    queue.join()
    print("Wczytywanie zajelo: ", (timer() - start) / 60)
    labels = list(list(zip(*storage))[0])
    images = list(list(zip(*storage))[1])
    return np.array(labels), np.array(images)


def preprocessing(labels, images):
    # Remove Asthma and LRTI
    images = np.delete(images, np.where((labels == 7) | (labels == 6))[0], axis=0)
    labels = np.delete(labels, np.where((labels == 7) | (labels == 6))[0], axis=0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=10)

    # Hot one encode the labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Format new data
    y_train = np.reshape(y_train, (y_train.shape[0], 6))
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 6))
    X_test = np.reshape(X_test, (X_test.shape[0], X_train.shape[1], 1))

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # # Data
    if not load:
        labels, images = data_points()
        if save:
            pickle.dump(labels, open('labels.p', 'wb'))
            pickle.dump(images, open('images.p', 'wb'))
    else:
        labels = pickle.load(open('labels.p', 'rb'))
        images = pickle.load(open('images.p', 'rb'))

    X_train, X_test, y_train, y_test = preprocessing(labels, images)

    # # Params
    activation_function = 'relu'
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=0.01,
                                                   patience=150,
                                                   verbose=0,
                                                   mode='auto',
                                                   restore_best_weights=True)
    log_dir = "logs/fit/" + activation_function
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model_checkpoint = keras.callbacks.ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss',
                                                       mode='min')

    # # Convolutional Neural Network
    model = Sequential()
    model.add(Conv1D(64, kernel_size=5, activation=activation_function, input_shape=(images.shape[1], 1)))

    model.add(Conv1D(128, kernel_size=5, activation=activation_function))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(256, kernel_size=5, activation=activation_function))
    model.add(MaxPooling1D(2))
    model.add(Flatten())

    model.add(Dense(512, activation=activation_function))
    model.add(Dropout(0.3))

    model.add(Dense(6, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=400, batch_size=200,
                        verbose=1, callbacks=[early_stopping, tensorboard_callback, model_checkpoint])

    # # Evaluation

    score = model.evaluate(X_test, y_test, batch_size=60, verbose=0)
    print('Accuracy: {0:.0%}'.format(score[1] / 1))
    print("Loss: %.4f\n" % score[0])

