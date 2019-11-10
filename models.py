import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa
import pickle
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=4, ncols=5, sharex=False,
                             sharey=True, figsize=(20,10))
    #fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(4):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i], size = 20)
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=4, ncols=5, sharex=False,
                             sharey=True, figsize=(20,10))
    #fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(4):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i], size = 20)
            axes[x,y].plot(freq, Y)
            axes[x,y].set_xlim(20, 4000)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=4, ncols=5, sharex=False,
                             sharey=True, figsize=(20,10))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(4):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=4, ncols=5, sharex=False,
                             sharey=True, figsize=(20,10))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(4):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)


df = pd.read_csv('freesound-audio-tagging/train_post_competition.csv')
df.set_index('fname', inplace=True)
for f in df.index:
  rate, signal = wavfile.read('freesound-audio-tagging/audio_train/' + f)
  df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y = 1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
      shadow = False, startangle=90)
ax.axis('equal')
plt.show()

df.reset_index(inplace=True)

signals = {}
fft = {}
fbank = {}
mfccs = {}

for c in classes:
    wav_file = df[df.label == c].iloc[0, 0]
    signal, rate = librosa.load('freesound-audio-tagging/audio_train/'+wav_file, sr=44100)
    mask = envelope(signal, rate, 0.0005)
    signal = signal[mask]
    signals[c] = signal
    fft[c] = calc_fft(signal,rate)

    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
    fbank[c] = bank
    mel = mfcc(signal[:rate], rate, numcep= 13, nfilt=26, nfft=1103).T
    mfccs[c] = mel

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join('freesound-audio-tagging/models', mode + '.model')
        self.p_path = os.path.join('freesound-audio-tagging/pickles', mode + '.p')
        self.min = -105.228205042938
        self.max = 102.59571471396549
    def changeMode(self, mode):
        self.mode = mode
        self.model_path = os.path.join('freesound-audio-tagging/models', mode + '.model')
        self.p_path = os.path.join('freesound-audio-tagging/pickles', mode + '.p')


df = pd.read_csv('instruments.csv')
df.set_index('fname', inplace=True)

for f in df2.index:
    rate, signal = wavfile.read('clean/'+f)
    df2.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df2.groupby(['label'])['length'].mean()

n_samples = 2 * int(df['length'].sum()/0.1)
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()

def check_data():
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

def build_rand_feat():
    #tmp = check_data()
    #if tmp:
    #    return tmp.data[0], tmp.data[1]
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in range(n_samples):
        rand_class = np.random.choice(class_dist.index, p = prob_dist)
        file = np.random.choice(df2[df2.label==rand_class].index)
        try:
            rate, wav = wavfile.read('clean/'+file)
        except:
            continue
        label = df2.at[file, 'label']
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate,
                       numcep = config.nfeat,
                        nfilt=config.nfilt, nfft=config.nfft)
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        y.append(classes.index(label))
    config.min = _min
    config.max = _max
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes = 41)
    config.data = (X, y)

    with open(config.p_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)

    return X, y


def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1,1),
                    padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1),
                    padding='same'))
    model.add(Conv2D(64, (3,3), activation='relu', strides=(1,1),
                    padding='same'))
    model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1),
                    padding='same'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(41, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                 optimizer = 'adam',
                 metrics=['acc'])
    return model

def get_recurrent_model():
    #shape of data for RNN is (n, time, feat)
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation ='relu')))
    model.add(TimeDistributed(Dense(32, activation ='relu')))
    model.add(TimeDistributed(Dense(16, activation ='relu')))
    model.add(TimeDistributed(Dense(8, activation ='relu')))
    model.add(Flatten())
    model.add(Dense(41, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                 optimizer = 'adam',
                 metrics=['acc'])
    return model


if config.mode == 'conv':
    #X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model=get_conv_model()

elif config.mode == 'time':
    #X, y = build_rand_feat
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()

class_weight = compute_class_weight('balanced',
                                   np.unique(y_flat),
                                   y_flat)

model.fit(X[0:100000], y[0:100000], epochs = 10, batch_size=32,
         shuffle=True,
         class_weight=class_weight)
