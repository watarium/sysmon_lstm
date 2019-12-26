import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
from keras import utils

print("You have TensorFlow version", tf.__version__)

# learning data
data = pd.read_csv("10_dlls.csv")
token_data = 'dlls_token.pickle'
encode_data = 'dlls_encode.pickle'
model_data = 'dlls_model.h5'
data = data.sample(frac=1)

tag_num = data['tags'].nunique()
data['tags'].value_counts()

max_words  = 10000
tokenizer = text.Tokenizer(num_words=max_words, char_level=False)

max_len = 50
tokenizer.fit_on_texts(data['data'])
sequences = tokenizer.texts_to_sequences(data['data'])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data_sec = pad_sequences(sequences, maxlen=max_len)

import pickle
# save the token data in the file
with open(token_data, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Split data into train and test
train_size = int(len(data_sec) * .8)
print ("The number of train data: %d" % train_size)
print ("The number of test data: %d" % (len(data_sec) - train_size))

x_train = data_sec[:train_size]
x_test = data_sec[train_size:]

train_tags = data['tags'][:train_size]
test_tags = data['tags'][train_size:]

# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

# save the encoder in the file
with open(encode_data, 'wb') as handle:
    pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# Build the model
model = Sequential()
model.add(Embedding(10000, 128, input_length=max_len))
# lstm=LSTM(32)
# print(lstm.units)
optimizer = Adam()
#optimizer = RMSprop()
model.add(LSTM(32))
model.add(Dense(tag_num, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=90,
                    verbose=1,
                    validation_data=(x_test, y_test))

# Evaluate the accuracy of our trained model
score = model.evaluate(x_test, y_test,
                       batch_size=32, verbose=1)
print('Loss of evaluation:', score[0])
print('Accuracy of evaluation:', score[1])

#save model to the file
model.save(model_data)

# Analyze the target sentense
# testdata = 'ntdll kernel32 KernelBase apphelp advapi32 msvcrt sechost rpcrt4 crypt32 ucrtbase msasn1 shlwapi combase ws2_32 bcryptprimitives gdi32 gdi32full user32 win32u cryptdll shell32 cfgmgr32 secur32 samlib windows.storage powrprof kernel.appcore SHCore ntdsapi profapi cryptsp cryptbase sspicli imm32 rsaenh bcrypt ncrypt ntasn1 vaultcli WinTypes wlanapi'
# testdata = [testdata]
# testdata_mat = tokenizer.texts_to_sequences(testdata)
# data_sec = pad_sequences(testdata_mat, maxlen=max_len)
# prediction = model.predict(np.array(data_sec))
# predicted_label = encoder.classes_[np.argmax(prediction)]
# print(predicted_label)
# print(prediction)

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# loss
def plot_history_loss(history):
    # Plot the loss in the history
    axL.plot(history.history['loss'],label="loss for training")
    axL.plot(history.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# acc
def plot_history_acc(history):
    # Plot the loss in the history
    axR.plot(history.history['acc'],label="loss for training")
    axR.plot(history.history['val_acc'],label="loss for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')

plot_history_loss(history)
plot_history_acc(history)
fig.savefig('./history.png')
plt.close()
