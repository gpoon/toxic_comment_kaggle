"""Following the CNN-based model from Text Understanding from Scratch.
https://arxiv.org/pdf/1502.01710.pdf

The interesting part of this model is that it works on character level vectors instead
of word level embeddings. An additional thing that could've been tested was using a
thesauraus to swap words with similar meanings and generate new training examples.
This was not attempted however because of memory constraints on my machine.

The test score of this was 0.074.
"""
from keras.callbacks import LearningRateScheduler
from keras.initializers import RandomNormal
from keras.layers import Conv1D, Embedding, Dense, Dropout, Flatten, Input, MaxPooling1D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras import regularizers
import numpy as np
import pandas as pd
import string

df = pd.read_csv('train.csv')
raw_comments = df.comment_text.values
classes = df.drop(columns=['id', 'comment_text']).columns
y = df[classes].values

characters = list(string.ascii_lowercase + string.digits + string.punctuation + '\n')
NUM_CHARS = len(characters)
char_vocab = {c: ind for ind, c in enumerate(characters)}
char_set = set(characters)
MAX_CHAR_LEN = 256   # Max num chars to consider for each frame, using Small Frame b/c OOM

# Custom Character Embedding
X_char = np.zeros((len(raw_comments), MAX_CHAR_LEN, NUM_CHARS), dtype=np.int8)
for i in range(len(raw_comments)):
    for j in range(len(raw_comments[i])):
        if j >= MAX_CHAR_LEN:
            break

        c = raw_comments[i][j].lower()
        if c in char_set:
	    X_char[i, j, char_vocab[c]] = 1

def lr_schedule(epoch):
    base = 0.01
    if epoch < 3:
        return base
    
    level = epoch / 3
    return base / (2. * level)

inp = Input(shape=(MAX_CHAR_LEN, NUM_CHARS))
x = Conv1D(MAX_CHAR_LEN, 7, activation='relu', kernel_initializer=RandomNormal(mean=0, stddev=0.05))(inp)
x = MaxPooling1D(pool_size=3)(x)
x = Conv1D(MAX_CHAR_LEN, 7, activation='relu', kernel_initializer=RandomNormal(mean=0, stddev=0.05))(x)
x = MaxPooling1D(pool_size=3)(x)
x = Conv1D(MAX_CHAR_LEN, 3, activation='relu', kernel_initializer=RandomNormal(mean=0, stddev=0.05))(x)
x = Conv1D(MAX_CHAR_LEN, 3, activation='relu', kernel_initializer=RandomNormal(mean=0, stddev=0.05))(x)
x = Conv1D(MAX_CHAR_LEN, 3, activation='relu', kernel_initializer=RandomNormal(mean=0, stddev=0.05))(x)
x = MaxPooling1D(pool_size=3)(x)
x = Flatten()(x)
x = Dense(1024, activation='relu', kernel_initializer=RandomNormal(mean=0, stddev=0.05))(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu', kernel_initializer=RandomNormal(mean=0, stddev=0.05))(x)
x = Dropout(0.5)(x)
x = Dense(len(classes), activation='sigmoid', kernel_initializer=RandomNormal(mean=0, stddev=0.05))(x)
cnn_char = Model(input=inp, output=x)

sgd = SGD(momentum=0.9)
cnn_char.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

lr_callback = LearningRateScheduler(lr_schedule)
cnn_char.fit(X_char, y, batch_size=128, epochs=30, callbacks=[lr_callback], verbose=1)

sub = pd.read_csv('test.csv')
test_comments = sub.comment_text.fillna('_na_').values
X_test = np.zeros((len(test_comments), MAX_CHAR_LEN, NUM_CHARS), dtype=np.int8)
for i in range(len(test_comments)):
    for j in range(len(test_comments[i])):
        if j >= MAX_CHAR_LEN:
            break

        c = test_comments[i][j].lower()
        if c in char_set:
	    X_test[i, j, char_vocab[c]] = 1

# Split up scoring because OOM
X_sub_parts = np.array_split(X_test, 10)
pred_sub_parts = map(lambda x: cnn_char.predict([x], batch_size=512, verbose=0), X_sub_parts)
pred_sub = np.concatenate(pred_sub_parts)
df_sub = pd.concat([sub.drop(columns=['comment_text']), pd.DataFrame(pred_sub, columns=classes)], axis=1)
df_sub.to_csv('cnn_char_paper.csv', index=False)
