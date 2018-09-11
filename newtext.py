import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import KFold
import pickle
import matplotlib.pyplot as plt
import random
import itertools
import tensorflow.contrib.keras as kr
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from cnn_model import *
from func import *
from config import Config
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


def sample(train_X, train_Y):
    train_len_GB = int((train_X.shape[0]) * 1)
    arr_A = random.sample(range(int(train_X.shape[0])), train_len_GB)
    train_index = arr_A[:int(train_len_GB*0.9)]
    val_index = arr_A[int(train_len_GB*0.9):]
    train_X_BU = train_X.loc[train_index, :]
    train_Y_BU = train_Y.loc[train_index]
    val_X_BU = train_X.loc[val_index, :]
    val_Y_BU = train_Y.loc[val_index]
    return train_X_BU, train_Y_BU, val_X_BU, val_Y_BU
def new_sample(train_X, train_Y):
    train_len_GB = int((train_X.shape[0]) * 1)
    arr_A = random.sample(range(int(train_X.shape[0])), train_len_GB)
    train_index = arr_A[:int(train_len_GB*0.95)]
    val_index = arr_A[int(train_len_GB*0.95):]
    train_X_BU = train_X[train_index]
    train_Y_BU = train_Y.loc[train_index]
    val_X_BU = train_X[val_index]
    val_Y_BU = train_Y.loc[val_index]
    return train_X_BU, train_Y_BU, val_X_BU, val_Y_BU
path0 = 'Data/result_train_embedding_word.pkl'
path1 = 'Data/result_test_embedding_word.pkl'
# path0 = 'Data/result_train_embedding_article.pkl'
# path1 = 'Data/result_test_embedding_article.pkl'
train = pd.read_csv('../new_data/data/train_set.csv')
test = pd.read_csv('../new_data/data/test_set.csv')
column = "word_seg"
train['word_seg'] = train['word_seg'].map(lambda t: np.array(t.split(' ')).astype(int))
train_x_I = train['word_seg'].values
test['word_seg'] = test['word_seg'].map(lambda t: np.array(t.split(' ')).astype(int))
test_x_I = test['word_seg'].values
# train['article'] = train['article'].map(lambda t: np.array(t.split(' ')).astype(int))
# train_x_I = train['article'].values
# test['article'] = test['article'].map(lambda t: np.array(t.split(' ')).astype(int))
# test_x_I = test['article'].values


y = ((train["class"])-1).astype(int)
kf = KFold(n_splits=5)
y = y.values
result = []
for train_inx, val_inx in kf.split(train_x_I):
    train_x_b = train_x_I[train_inx]
    y_b = y[train_inx]
    train_x = train_x_b[:int((train_x_b.shape[0])*0.95)]
    train_y = y_b[:int((train_x_b.shape[0])*0.95)]
    val_x = train_x_b[int((train_x_b.shape[0])*0.95):]
    val_y = y_b[int((train_x_b.shape[0])*0.95):]
    # train_x = train_x_b[:int((train_x_b.shape[0])*0.01)]
    # train_y = y_b[:int((train_x_b.shape[0])*0.01)]
    # val_x = train_x_b[int((train_x_b.shape[0])*0.99):]
    # val_y = y_b[int((train_x_b.shape[0])*0.99):]
    MODEL = train_network(train_x, train_y, val_x, val_y)
    generate_x = train_x_I[val_inx]
    generate_x = kr.preprocessing.sequence.pad_sequences(generate_x, Config.seq_length, padding='post', truncating='post')
    result.append(generate_result(generate_x, MODEL))
    del MODEL
result = list(itertools.chain.from_iterable(result))
result_dir = path0
pickle.dump(result, open(result_dir, "wb"))
train_x = train_x_I[:int(len(train_x_I)*0.99)]
val_x = train_x_I[int(len(train_x_I)*0.99):]
train_y = y[:int(len(train_x_I)*0.99)]
val_y = y[int(len(train_x_I)*0.99):]
MODEL = train_network(train_x, train_y, val_x, val_y)
generate_test_x = kr.preprocessing.sequence.pad_sequences(test_x_I, Config.seq_length, padding='post', truncating='post')
result = generate_result(generate_test_x, MODEL)
result_dir = path1
pickle.dump(result, open(result_dir, "wb"))

# #vec = TfidfVectorizer(ngram_range=(1,2), stop_words=wordseg_most_common[:10],use_idf=1,smooth_idf=1, sublinear_tf=1)
# y = ((train["class"])-1).astype(int)
# print("fitting Start !!!")
# train_data_dir = 'Data/data.pkl'
# test_data_dir = 'Data/data_test.pkl'
# if not os.path.exists(train_data_dir):
#     trn_term_doc = vec.fit_transform(train[column])
#     tes_term_doc = vec.transform(test[column])
#     pickle.dump(trn_term_doc, open(train_data_dir, "wb"))
#     pickle.dump(tes_term_doc, open(test_data_dir, "wb"))
# else:
#     trn_term_doc = pickle.load(open(train_data_dir,"rb"))
# train_x_dir = 'Data/train_x.pkl'
# train_y_dir = 'Data/train_y.pkl'
# val_x_dir = 'Data/val_x.pkl'
# val_y_dir = 'Data/val_y.pkl'
# if not os.path.exists(train_x_dir):
#     print("Saving Train and Val Data")
#     train_x, train_y, val_x, val_y = new_sample(trn_term_doc, y)
#     pickle.dump(train_x, open(train_x_dir, "wb"))
#     pickle.dump(train_y, open(train_y_dir, "wb"))
#     pickle.dump(val_x, open(val_x_dir, "wb"))
#     pickle.dump(val_y, open(val_y_dir, "wb"))
# else:
#     print("Loading Train and Val Data")
#     train_x = pickle.load(open(train_x_dir,"rb"))
#     train_y = pickle.load(open(train_y_dir, "rb"))
#     val_x = pickle.load(open(val_x_dir, "rb"))
#     val_y = pickle.load(open(val_y_dir, "rb"))
# config = TCNNConfig()
# print('tfidf length is {0}'.format(trn_term_doc.shape[1]))
# config.seq_length = trn_term_doc.shape[1]
# train_network(train_x, train_y, val_x, val_y, config)
