# -*- coding: utf-8 -*-#
"""
@author:Galen
@file: xunfei_classify.py
@time: 2019/06/18
@description:
中文文本分类 xunfei_classify 最优


步骤：
1. 生成 Tokenizer

参考资料
https://blog.csdn.net/asialee_bird/article/details/88813385
tokenizer:
https://www.cnblogs.com/bymo/p/9675654.html
模型结构
https://blog.csdn.net/asialee_bird/article/details/88813385
https://www.cnblogs.com/bymo/p/9675654.html
"""
import argparse
import os
import pickle
import re
from datetime import datetime

import numpy as np
import pandas as pd
import tqdm
from keras import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv1D, MaxPool1D, Dense, Flatten, concatenate, Embedding, Dropout
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model, to_categorical

parse = argparse.ArgumentParser()
parse.add_argument('--data', '-d', type=str, default='data/xfyun')
parse.add_argument('--name', '-n', type=str, default='xunfei')  #
parse.add_argument('--model', '-m', type=str, default='train')  # train', 'test'
parse.add_argument('--sentence', '-s', type=int, default=1000)  # 句子长度. 1000 > 1500
parse.add_argument('--vocab', '-v', type=int, default=6000)  # 词长度。
parse.add_argument('--embedding', '-e', type=int, default=64)  # 词向量维度。 64 > 128
args = parse.parse_args()

data_path = args.data
WEIGHTS_IDR = './weight/xunfei_classify'

# 创建权重文件夹
if not os.path.exists(WEIGHTS_IDR):
    os.makedirs(WEIGHTS_IDR)
    print('[Info] 文件夹 "%s" 不存在, 创建文件夹.' % WEIGHTS_IDR)

model_name = "{0}.xunfei_{1}_{2}_{3}".format(str(args.name), str(args.sentence), str(args.vocab), str(args.embedding))
print(model_name)
WEIGHTS_FILE = os.path.join(WEIGHTS_IDR, "{0}.hdf5".format(model_name))
MODEL_FILE = os.path.join(WEIGHTS_IDR, "{0}.h5".format(model_name))

SENTENCE_LEN = args.sentence  # 句子长度 6000 1500
VOCAB_LEN = args.vocab  # 词长度
EMBEDDING_DIM = args.embedding  # 词向量维度


def load_cate():
    cate_path = "data/xfyun/apptype_count.txt"
    with open(cate_path, 'r')as f:
        data_list = f.readlines()
        return [x.split("\t")[0] for x in data_list]


BATCH_SIZE = 64
CATEGORIES = load_cate()
NUM_CATEGORY = len(CATEGORIES)
tokenizer_path = "tokenizer_{0}_{1}.pickle".format(args.name, args.vocab)
train_file = os.path.join(data_path, '{0}.train.txt'.format(args.name))
val_file = os.path.join(data_path, '{0}.val.txt'.format(args.name))  # 3000
test_file = os.path.join(data_path, 'app_desc.dat')

TOTAL_EPOCHS = 20
STEPS_PER_EPOCH = int(28134 // BATCH_SIZE)
VALIDATION_STEPS = int(300 // BATCH_SIZE)


def clean_multiple_spaces(content):
    """
    清理非多个空格
    :param content: 
    :return: 
    """

    # content = re.sub(r'[^\u4e00-\u9fa5]+', ' ', content) # Eliminate Chinese characters
    return re.sub('[\r\n\t ]+', ' ', content).replace('\xa0', '').strip()


def read_data_file(path):
    lines = open(path, 'r', encoding='utf-8').readlines()
    x_list = []
    y_list = []
    for line in tqdm.tqdm(lines):
        rows = line.split('\t')
        if len(rows) >= 2:
            y_list.append(rows[0].strip())
            x_list.append(list(clean_multiple_spaces(' '.join(rows[1:]))))
        else:
            pass
            # print(rows)
    return x_list, y_list


def tokenizer_data():
    """
    将语料进行tokenrizer
    取语料的 90 % 长度作为最大长度值。
    :return:
    """
    if os.path.exists(tokenizer_path):
        # loading
        print("loading tokenizer data...")
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        print("calculate tokenizer data...")
        test_x, test_y = read_data_file(test_file)
        train_x, train_y = read_data_file(train_file)
        val_x, val_y = read_data_file(val_file)
        data_all = test_x + train_x + val_x
        # 或的0.90 左右文本长度
        sequence_length = sorted([len(x) for x in data_all])[int(0.95 * len(data_all))]
        print("sequence_length 90 is : {0}".format(str(sequence_length)))
        tokenizer = Tokenizer(VOCAB_LEN)
        tokenizer.fit_on_texts(data_all)
        # loading
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer


def make_callbacks(weights_file):
    checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

    #  不再优化时候，调整学习率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=1, mode='auto', factor=0.8)

    # 当验证集的loss不再下降时，中断训练 patience=2 两个 Epoch 停止
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # all the goodies
    return [reduce_lr, checkpoint, early_stopping]


def xunfei_classify_model(sentence_length=SENTENCE_LEN, vocab_len=VOCAB_LEN, embedding_dim=EMBEDDING_DIM,
                          model_img_path=None,
                          embedding_matrix=None):
    """
    TextCNN:
        1. embedding layers,
        2.convolution layer,
        3.max-pooling,
        4.softmax layer.
    :param sentence_length:句子大小
    :param vocab_len: 文本中词汇表大小
    :param embedding_dim: 词向量空间大小
    :param model_img_path:
    :param embedding_matrix:
    :return:
    """
    x_input = Input(shape=(sentence_length,))

    if embedding_matrix is None:
        x_emb = Embedding(input_dim=vocab_len + 1, output_dim=embedding_dim, input_length=sentence_length)(x_input)
    else:
        x_emb = Embedding(input_dim=vocab_len + 1, output_dim=embedding_dim, input_length=sentence_length,
                          weights=[embedding_matrix], trainable=True)(x_input)

    # 多层卷积核
    pool_output = []
    kernel_sizes = [3, 4, 5]
    for kernel_size in kernel_sizes:
        c = Conv1D(filters=256, kernel_size=kernel_size, strides=1)(x_emb)
        p = MaxPool1D(pool_size=int(c.shape[1]))(c)
        pool_output.append(p)
    # 合并三个模型的输出向量
    pool_output = concatenate([p for p in pool_output])

    x_flatten = Flatten()(pool_output)
    drop = Dropout(0.5)(x_flatten)
    y = Dense(NUM_CATEGORY, activation='softmax')(drop)
    model = Model([x_input], outputs=[y])
    if model_img_path:
        plot_model(model, to_file=model_img_path, show_shapes=True, show_layer_names=False)
    # model.summary()
    return model


def process_file(filename, tokenizer, cate_id, max_length=SENTENCE_LEN):
    """将文件转换为id表示"""
    contents, labels = read_data_file(filename)
    label_id = [cate_id[x] for x in labels]
    # 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
    data_id = tokenizer.texts_to_sequences(contents)
    # 使用keras提供的pad_sequences来将文本pad为固定长度 句子不够长则补0
    x_pad = pad_sequences(data_id, max_length)
    # label进行one-hot处理 label进行labelEncoder之后变为0-9个数才能进行one-hot，且由于10个类别，则每个label的维度大小为10。
    y_pad = to_categorical(label_id, num_classes=len(cate_id))  # 将标签转换为one-hot表示
    return x_pad, y_pad


def predict_file(filename, tokenizer, max_length=SENTENCE_LEN):
    """将文件转换为id表示"""
    contents, labels = read_data_file(filename)
    # 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
    data_id = tokenizer.texts_to_sequences(contents)
    # 使用keras提供的pad_sequences来将文本pad为固定长度 句子不够长则补0
    x_pad = pad_sequences(data_id, max_length)
    # label进行one-hot处理 label进行labelEncoder之后变为0-9个数才能进行one-hot，且由于10个类别，则每个label的维度大小为10。
    return x_pad, labels


def batch_iter(x, y, batch_size=BATCH_SIZE):
    """
    生成批次数据
    :param x:
    :param y:
    :param batch_size:
    :return:
    """
    while 1:
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1
        # shuffle与permutation都是对原来的数组进行重新洗牌 permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组。
        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]
        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def train():
    # 加载数据
    print("Loading training and validation data...")
    x_train, y_train = process_file(train_file, tokenize, cate_to_id, SENTENCE_LEN)
    x_val, y_val = process_file(val_file, tokenize, cate_to_id, SENTENCE_LEN)
    train_batch = batch_iter(x_train, y_train)
    val_batch = batch_iter(x_val, y_val)
    model = xunfei_classify_model()
    if os.path.exists(WEIGHTS_FILE):
        print("loading ", WEIGHTS_FILE)
        model.load_weights(WEIGHTS_FILE, by_name=True)
    callbacks_list = make_callbacks(WEIGHTS_FILE)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit_generator(train_batch,
                        epochs=TOTAL_EPOCHS,
                        callbacks=callbacks_list,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=val_batch,
                        validation_steps=VALIDATION_STEPS,
                        verbose=1,  # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
                        )
    # Save it for later
    print('Saving Model')
    # Keras模型和权重保存在一个HDF5文件中
    model.save(MODEL_FILE, include_optimizer=True)


def sort_dict(dict_words, reverse=True, site=1):
    """
    字典排序
    reverse: False 升序，True降序
    site: 0 第一个元素，1是第二个元素
    :param dict_words:
    :return:
    """
    keys = dict_words.keys()
    values = dict_words.values()
    list_one = [(key, val) for key, val in zip(keys, values)]
    list_sort = sorted(list_one, key=lambda x: x[site], reverse=reverse)
    return list_sort


def get_top2(data_list):
    result_label_1 = []
    result_label_2 = []
    for data in data_list.tolist():
        cate_score_list = zip(range(len(CATEGORIES)), data)
        list_sort = sorted(cate_score_list, key=lambda x: x[1], reverse=True)
        # print('list_sort', list_sort[0:10])
        result_label_1.append(list_sort[0][0])
        result_label_2.append(list_sort[0][1])
    return result_label_1, result_label_2


def predict():
    print("Loading desc data...")
    x_test, y_name = predict_file(test_file, tokenize, SENTENCE_LEN)
    model = load_model(MODEL_FILE)
    result = model.predict(x_test)  # 预测样本属于每个类别的概率
    # print("result", result)
    label_1, label_2 = get_top2(result)
    label_1_predict_name = [id_to_cate.get(x, "140901") for x in label_1]
    label_2_predict_name = [id_to_cate.get(x, "140206") for x in label_2]
    df_sub = pd.concat([pd.Series(y_name), pd.Series(label_1_predict_name), pd.Series(label_2_predict_name)], axis=1)
    df_sub.columns = ['id', 'label1', 'label2']
    df_sub.to_csv('pre_test_{}.csv'.format(datetime.now().strftime('%m%d_%H%M%S')), sep=',', index=False)


if __name__ == '__main__':
    if args.model not in ['train', 'predict']:
        raise ValueError("""usage: python run_cnn.py [train / predict]""")
    cate_to_id = dict(zip(CATEGORIES, range(len(CATEGORIES))))
    id_to_cate = dict(zip(range(len(CATEGORIES)), CATEGORIES))
    # 加载数据

    tokenize = tokenizer_data()
    if args.model == 'train':
        train()
    else:
        predict()
