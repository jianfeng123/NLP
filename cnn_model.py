import tensorflow as tf
import os
from datetime import timedelta
import numpy as np
import time
import sys
import tensorflow.contrib.keras as kr
from itertools import chain
from config import Config
import itertools
from func import *
# def compute_f1_score(label_list_top5,eval_y):
#     """
#     compoute f1_score.
#     :param logits: [batch_size,label_size]
#     :param evalY: [batch_size,label_size]
#     :return:
#     """
#     num_correct_label=0
#     eval_y_short=get_target_label_short(eval_y)
#     for label_predict in label_list_top5:
#         if label_predict in eval_y_short:
#             num_correct_label=num_correct_label+1
#     #P@5=Precision@5
#     num_labels_predicted=len(label_list_top5)
#     all_real_labels=len(eval_y_short)
#     p_5=num_correct_label/num_labels_predicted
#     #R@5=Recall@5
#     r_5=num_correct_label/all_real_labels
#     f1_score=2.0*p_5*r_5/(p_5+r_5+0.000001)
#     return f1_score,p_5,r_5

def _highway_layer(input_, size, num_layers=1):
    """
    Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    for idx in range(num_layers):
        fc0 = tf.layers.dense(input_, size, activation=tf.nn.relu)
        fc1 = tf.layers.dense(fc0, size, activation=tf.sigmoid)
        output = fc0 * fc1 + (1. - fc1) * input_
        input_ = output
    return output

class TextCNN(object):
    """文本分类，CNN模型"""
    def __init__(self, saving_loading = False):
        # 三个待输入的数据
        tf.reset_default_graph()
        self.input_x = tf.placeholder(tf.int32, [None, Config.seq_length], name='input_x')
        # self.input_x = tf.placeholder(tf.float32, [None, self.Config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, Config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        if saving_loading == True:
            self.save_dir = 'checkpoints/textcnn'
            self.save_path = os.path.join(self.save_dir, 'best_validation')
            self.saver = load_model(self.session, self.save_dir)
        self.saving_or_loading = saving_loading
    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.name_scope('embedding'):
            embedding = tf.get_variable('embedding', [Config.vocab_size, Config.embedding_dim])
            embedding_outputs = tf.nn.embedding_lookup(embedding, self.input_x)
        pooled_outputs = []
        for filter_size in Config.kernel_size:
            with tf.name_scope('conv_filter{0}'.format(filter_size)):
                conv1 = tf.layers.conv1d(embedding_outputs, 128,kernel_size=filter_size,strides=1, padding='VALID')
#                 conv_bn_output = tf.layers.batch_normalization(conv1)
                conv_re_output = tf.nn.relu(conv1)
            with tf.name_scope("pool_filter{0}".format(filter_size)):
                # pooled = tf.layers.max_pooling2d(conv_re_output,pool_size=[self.Config.seq_length-filter_size+1, 1],strides=1, padding='VALID', name="pool")
                pooled0 = tf.reduce_max(conv_re_output, reduction_indices=[1])
                pooled1 = tf.reduce_mean(conv_re_output, reduction_indices=[1])
                pooled_combine = tf.concat([pooled0, pooled1], axis=-1)
            pooled_outputs.append(pooled_combine)
        h_pool = tf.concat(pooled_outputs, axis=1)
        # h_pool_fl = tf.layers.flatten(h_pool)
#         with tf.name_scope('fc'):
#             fc0 = tf.layers.dense(h_pool, 512, name='fc0',kernel_regularizer=tf.keras.regularizers.l2(0.01))
#             fc0 = tf.contrib.layers.dropout(fc0, self.keep_prob)
#             fc0_out = tf.nn.relu(fc0)
#         with tf.name_scope("highway"):
#             highway = _highway_layer(h_pool, h_pool.get_shape()[1], num_layers=2)
            # fc_input1 = tf.layers.dense(fc_input0, 256, name='fc1', kernel_regularizer=tf.keras.regularizers.l2(0.01),
            #                             activation=tf.nn.relu)
            # fc_input0 = tf.layers.batch_normalization(fc_input0)
        #     gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
#             fc = tf.layers.dense(h_pool, 64, name='fc2')
#             fc = tf.contrib.layers.dropout(fc, self.keep_prob)
#             fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(h_pool, Config.num_classes, name='fc3')
            self.soft = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(self.soft, 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=Config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            self.label = tf.argmax(self.input_y, 1)
# a = TCNNConfig()
# b = TextCNN(a)

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def process(X, Y, max_length=700, class_count=19):
    """将dataframeX,Y的训练数据提取出来"""
    # X = X['word_seg'].values
    # Y = Y.values
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(X, max_length,padding='post',truncating='post')
    y_pad = kr.utils.to_categorical(Y, num_classes=class_count)  # 将标签转换为one-hot表示
    return x_pad, y_pad
# def process(X, Y, max_length=600, class_count=19):
#     """将dataframeX,Y的训练数据提取出来"""
#     # X = X['word_seg'].values
#     Y = Y
#     # 使用keras提供的pad_sequences来将文本pad为固定长度
#     # x_pad = kr.preprocessing.sequence.pad_sequences(X, max_length,padding='post',truncating='post')
#     y_pad = kr.utils.to_categorical(Y, num_classes=class_count)  # 将标签转换为one-hot表示
#     return X, y_pad

def load_model(sess, path):
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old weights!")
    return saver

def train_network(train_x, train_y, val_x, val_y, saving_loading = False):
    model = TextCNN()
    # print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    # tensorboard_dir = 'tensorboard/textcnn'
    # if not os.path.exists(tensorboard_dir):
    #     os.makedirs(tensorboard_dir)
    #
    # tf.summary.scalar("loss", model.loss)
    # tf.summary.scalar("accuracy", model.acc)
    # merged_summary = tf.summary.merge_all()
    # writer = tf.summary.FileWriter(tensorboard_dir)
    res_model = model
    # 配置 Saver 或者导入weight
    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process(train_x, train_y, Config.seq_length)
    x_val, y_val = process(val_x, val_y, Config.seq_length)
    # x_train, y_train = train_x, train_y
    # x_val, y_val = val_x, val_y
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    # writer.add_graph(session.graph)
    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_F1 = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 300000  # 如果超过1000轮未提升，提前结束训练
    flag = False
    for epoch in range(Config.num_epochs):
        print('Epoch:', epoch + 1)
        # tf.reset_default_graph()  # 重置默认图
        # graph = tf.Graph()  # 新建空白图
        # with graph.as_default() as g:  # 将新建的图作为默认图
        #     with tf.Session(graph=g) as session:  # Session  在新建的图中运行

        # 需要运行的代码放这里，每次运行都会使用新的图
        batch_train = batch_iter(x_train, y_train, Config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, Config.dropout_keep_prob, model)
            # if total_batch % Config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                # s = session.run(merged_summary, feed_dict=feed_dict)
                # writer.add_summary(s, total_batch)
            if total_batch % Config.save_per_batch == 0:
                mes = "Iter: {0:>6}, Time: {1}"
                time_dif = get_time_dif(start_time)
                sys.stdout.flush()
                print(mes.format(total_batch, time_dif))
            if total_batch % Config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = model.session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, F1, ACC = evaluate(x_val, y_val, model)  # todo
                for i in range(1):
                    print(" * ")
                    sys.stdout.flush()
                    time.sleep(1)
                if F1 > best_F1:
                    # 保存最好结果
                    best_F1 = F1
                    last_improved = total_batch
                    if model.saving_or_loading == True:
                        model.saver.save(sess=model.session, save_path=model.save_path)
                    res_model = model
                    improved_str = '*'
                    sys.stdout.flush()
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val F1: {4:>7.2%},Val Acc: {5:>7.2%} Time: {6} {7}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, F1, ACC, time_dif, improved_str))

            model.session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break
    return res_model
