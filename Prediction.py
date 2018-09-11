from __future__ import print_function
import pandas as pd
import os
import tensorflow as tf
import tensorflow.contrib.keras as kr
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from cnn_model import TCNNConfig, TextCNN
import pickle
import sys
import numpy as np
import itertools
categories = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
base_dir = 'data/cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

class CnnModel:
    def __init__(self, LENGTH):
        self.config = TCNNConfig()
        self.config.seq_length = LENGTH
        self.model = TextCNN(self.config)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        #content = unicode(message)
        #for x in content[:10]:
        #    print(x)
        feed_dict = {
            self.model.input_x: message.toarray(),
            self.model.keep_prob: 1.0
        }
        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return categories[y_pred_cls[0]]

def batch_iter(x, batch_size=128):
    """生成批次数据"""
    data_len = x.shape[0]
    num_batch = int((data_len - 1) / batch_size) + 1
    indices = np.arange(data_len)
    x_shuffle = x[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id]
def evaluate(x_, MODEL):
    """评估在某一数据上的准确率和损失"""
    data_len = x_.shape[0]
    batch_eval = batch_iter(x_, 128)
    result = []
    x = 0
    for x_batch in batch_eval:
        print(x)
        if x%100 == 0:
            sys.stdout.flush()
        feed_dict = {
            MODEL.model.input_x: x_batch.toarray(),
            MODEL.model.keep_prob: 1.0
        }
        y_pred_cls = MODEL.session.run(MODEL.model.y_pred_cls, feed_dict=feed_dict)
        result.append(y_pred_cls)
        x = x + 1
    result = list(itertools.chain.from_iterable(result))
    result = np.array(result).reshape([data_len])
    return result

train = pd.read_csv('train_set.csv')
test = pd.read_csv('test_set.csv')


del train['article']
column = "word_seg"
vec = TfidfVectorizer(ngram_range=(1,2), min_df=8, max_df=0.4,use_idf=1,smooth_idf=1, sublinear_tf=1)
data_dir = 'Data/data_test.pkl'
if not os.path.exists(data_dir):
    trn_term_doc = vec.fit_transform(train[column])
    tes_term_doc = vec.transform(test[column])
    pickle.dump(tes_term_doc, open(data_dir, "wb"))
else:
    tes_term_doc = pickle.load(open(data_dir,"rb"))
cnn_model = CnnModel(tes_term_doc.shape[1])
result = evaluate(tes_term_doc, cnn_model)

i=0
fid0=open('result/result_test_cnn.csv','w')
fid0.write("id,class"+"\n")
for item in result:
    fid0.write(str(i)+","+str(item+1)+"\n")
    i=i+1
fid0.close()
