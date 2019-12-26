from keras.models import load_model
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import pickle

model = load_model('noise_gutenberg_model.h5', compile=False)
with open('noise_gutenberg_token.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('noise_gutenberg_encode.pickle', 'rb') as handle:
    encoder = pickle.load(handle)
data = pd.read_csv("gutenberg_test.csv")
eval_data = data['data']
eval_tag = data['tags']

max_len = 50
testdata_mat = tokenizer.texts_to_sequences(eval_data)
data_sec = pad_sequences(testdata_mat, maxlen=max_len)
prediction = model.predict(np.array(data_sec))

correct = 0
failure = 0

for i in range(len(prediction)):
    predicted_label = encoder.classes_[np.argmax(prediction[i])]
    if predicted_label == eval_tag[i]:
        correct += 1
    else:
        failure += 1
n = correct + failure

print('N: ' + str(n) + '\n' + 'correct rate: ' + '{:.2f}'.format(correct / n) + ' failure_rate: ' + '{:.2f}'.format(failure / n))
