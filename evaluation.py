from keras.models import load_model
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import pickle

model = load_model('dlls_model.h5', compile=False)
with open('dlls_token.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('dlls_encode.pickle', 'rb') as handle:
    encoder = pickle.load(handle)
data = pd.read_csv("5_dlls_test.csv")
#data = pd.read_csv("5_dlls_test.csv")
eval_data = data['data']
eval_tag = data['tags']

max_len = 50
testdata_mat = tokenizer.texts_to_sequences(eval_data)
data_sec = pad_sequences(testdata_mat, maxlen=max_len)
prediction = model.predict(np.array(data_sec))

fn = 0
fp = 0
tn = 0
tp = 0

for i in range(len(prediction)):
    predicted_label = encoder.classes_[np.argmax(prediction[i])]
    if predicted_label == eval_tag[i]:
        if eval_tag[i] == 'normal':
            tn += 1
        else:
            tp += 1
    else:
        if eval_tag[i] == 'normal':
            fn += 1
            print(i+1)
            print('True value: ' +str(eval_tag[i]))
            print('Predicted value: ' + str(predicted_label))
            print(eval_data[i])
        else:
            fp += 1
            print(i+1)
            print('True value: ' +str(eval_tag[i]))
            print('Predicted value: ' + str(predicted_label))
            print(eval_data[i])

recall = tp / (tp + fp + 1e-09)
precision = tp / (tp + fn + 1e-09)
accuracy = (tp + tn) / (fn + fp + tn + tp + 1e-09)

print('N: ' + str(i) + '\n' + 'Recall: ' + '{:.2f}'.format(recall) + ' Precision: ' + '{:.2f}'.format(precision) + ' Accuracy: ' + '{:.2f}'.format(accuracy))
