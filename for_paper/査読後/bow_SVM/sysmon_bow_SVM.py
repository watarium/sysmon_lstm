import itertools

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np

print("You have TensorFlow version", tf.__version__)


# learning data
data = pd.read_csv("00_train.csv")
lsa_data = 'bow_svm_dlls_lsa.pickle'
fit_data = 'bow_svm_dlls_fit.pickle'
model_data = 'bow_svm_dlls_model.pickle'
data = data.sample(frac=1)

data['tags'].value_counts()

# Split data into train and test
train_size = int(len(data) * .8)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(data) - train_size))

train_data = data['data'][:train_size]
train_tags = data['tags'][:train_size]

test_data = data['data'][train_size:]
test_tags = data['tags'][train_size:]

# CountVectorizer
# cvec = CountVectorizer(min_df=0.24, max_df=0.76)
cvec = CountVectorizer()

# ベクトル化
bow_vector_fit = cvec.fit(train_data)
bow_vector = cvec.fit_transform(train_data)

# 2-2-1.パラメータの調整
# list_n_comp = [5,10,50,100,500,1000] # 特徴量を何個に削減するか、というパラメータです。できるだけ情報量を欠損しないで、かつ次元数は少なくしたいですね。
# for i in list_n_comp:
#     lsa = TruncatedSVD(n_components=i,n_iter=5, random_state = 0)
#     lsa.fit(tfv_vector)
#     tfv_vector_lsa = lsa.transform(tfv_vector)
#     print('次元削減後の特徴量が{0}の時の説明できる分散の割合合計は{1}です'.format(i,round((sum(lsa.explained_variance_ratio_)),2)))

# 2-2-2.次元削減した状態のデータを作成
# 上記で確認した「n_components」に指定した上で、次元削減（特徴抽出）を行う
# lsa = TruncatedSVD(n_components=500,n_iter=5, random_state = 0) # 今回は次元数を500に指定
# lsa.fit(tfv_vector)
# tfv_vector_lsa = lsa.transform(tfv_vector)

# clf = GridSearchCV(svm.SVC(), param(), scoring="accuracy",cv=5, n_jobs=-1)
# clf.fit(tfv_vector_lsa, train_tags) # 3-1-4.学習
# for params, mean_score, all_scores in clf.grid_scores_:
#         print ("{0},精度:{1} ,標準誤差=(+/- {2}) ".format(params, round((mean_score),3), round((all_scores.std() / 2),3))) # 各パラメータごとの精度を確認

# for rate in ['05', '10', '15', '20', '25', '30', '35', '40']:
for rate in ['05']:
    print('---------------noise rate: '+str(rate)+'--------------')
    for n in range(1):
        for C in [100]:
            for gamma in [0.5]:
                model = svm.SVC(C = C, gamma = gamma, kernel = 'rbf', class_weight='balanced')
                clf = model.fit(bow_vector, train_tags)

                data = pd.read_csv('/Users/watarium/PycharmProjects/sysmon_lstm/for_paper/20191223_evidence_noise/noise_eval_log/add_reduce'+str(rate)+'_test_'+str(n)+'.csv')
                # data = pd.read_csv('/Users/watarium/PycharmProjects/sysmon_lstm/for_paper/20191223_evidence_noise/noise_eval_log/add_reduce' + str(rate) + '_test_' + str(n) + '.csv')
                # data = pd.read_csv('/Users/watarium/PycharmProjects/sysmon_lstm/for_paper/20191223_evidence_noise/noise_eval_log/reduction' + str(rate) + '_test_' + str(n) + '.csv')
                # data = pd.read_csv('00_test.csv')
                eval_data = data['data']
                eval_tag = data['tags']

                # ベクトル化
                bow_vector_test = cvec.transform(eval_data)

                # 予測
                prediction = clf.predict(bow_vector_test)

                fn = 0
                fp = 0
                tn = 0
                tp = 0

                for i in range(len(prediction)):
                    predicted_label = prediction[i]
                    if predicted_label == eval_tag[i]:
                        if eval_tag[i] == 'normal':
                            tn += 1
                        else:
                            tp += 1
                    else:
                        if eval_tag[i] == 'normal':
                            fn += 1
                            print(i + 1)
                            print('True value: ' + str(eval_tag[i]))
                            print('Predicted value: ' + str(predicted_label))
                            print(eval_data[i])
                        else:
                            fp += 1
                            print(i + 1)
                            print('True value: ' + str(eval_tag[i]))
                            print('Predicted value: ' + str(predicted_label))
                            print(eval_data[i])

                recall = tp / (tp + fp + 1e-09)
                precision = tp / (tp + fn + 1e-09)
                accuracy = (tp + tn) / (fn + fp + tn + tp + 1e-09)

                print('C: ' + str(C) + ', gamma: ' + str(gamma))
                print('N: ' + str(i) + '\n' + 'Recall: ' + '{:.2f}'.format(recall) + ' Precision: ' + '{:.2f}'.format(
                    precision) + ' Accuracy: ' + '{:.2f}'.format(accuracy))
