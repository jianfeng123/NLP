class TCNNConfig(object):
    """CNN配置参数"""
    embedding_dim = 64  # 词向量维度
    seq_length = 1000  # 序列长度
    num_classes = 19  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = [2,3,4,5]  # 卷积核尺寸
    vocab_size = 1279999  # 词汇表达小
    hidden_dim = 256  # 全连接层神经元
    # hidden_dim1 = 128
    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 2e-3  # 学习率
    lstm_hidden_size = 256
    batch_size = 64  # 每批训练大小
    num_epochs = 50  # 总迭代轮次

    print_per_batch = 500  # 每多少轮输出一次结果
    save_per_batch = 50  # 每多少轮存入tensorboard

Config = TCNNConfig()