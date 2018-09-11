import numpy as np
import sys
from itertools import chain
import itertools
def feed_data(x_batch, y_batch, keep_prob, model):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = x.shape[0]
    num_batch = int((data_len - 1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def evaluate(x_, y_, model):
    """评估在某一数据上的准确率和损失"""
    data_len = x_.shape[0]
    batch_eval = batch_iter(x_, y_, 64)
    total_loss = 0.0
    accuracy = []
    recall = []
    sys.stdout.flush()
    for x_batch, y_batch in batch_eval:
        batch_len = x_batch.shape[0]
        feed_dict = feed_data(x_batch, y_batch, 1.0, model)
        sys.stdout.flush()
        loss, A, B = model.session.run([model.loss, model.y_pred_cls, model.label], feed_dict=feed_dict)
        accuracy.append(A)
        recall.append(B)
        total_loss += loss * batch_len
    accuracy = list(chain.from_iterable(accuracy))
    recall = list(chain.from_iterable(recall))
    ACC = []
    REC = []
    REAL = []
    for i in range(19):
        ACC.append(accuracy.count(i))
        REC.append(recall.count(i))
        Index = [j for j,value in enumerate(accuracy) if value == i]
        REAL.append([int(accuracy[j]==recall[j]) for j in Index].count(1))
    F1 = 0
    AP = 0
    for i in range(19):
        if ACC[i] == 0:
            p = 0
        else:
            p = float(REAL[i]) / float(ACC[i])
        if REC[i] == 0:
            r = 0
        else:
            r = float(REAL[i]) / float(REC[i])
        if p==0 and r == 0:
            F1 += 0
        else:
            AP += p
            F1 += p * r * 2 / (p + r)
    return total_loss / data_len, F1/19., AP/19
def g_batch_iter(x, batch_size=128):
    """生成批次数据"""
    data_len = x.shape[0]
    num_batch = int((data_len - 1) / batch_size) + 1
    indices = np.arange(data_len)
    x_shuffle = x[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id]

def generate_result(x_, MODEL):
    """评估在某一数据上的准确率和损失"""
    data_len = x_.shape[0]
    batch_eval = batch_iter(x_, 128)
    result = []
    x = 0
    for x_batch in batch_eval:
        print(x)
        if x % 100 == 0:
            sys.stdout.flush()
        feed_dict = {
            MODEL.input_x: x_batch,
            MODEL.keep_prob: 1.0
        }
        pred_cls_soft = MODEL.session.run(MODEL.soft, feed_dict=feed_dict)
        result.append(pred_cls_soft)
        x = x + 1
    result = list(itertools.chain.from_iterable(result))
    # result = np.array(result).reshape([data_len])
    return result
