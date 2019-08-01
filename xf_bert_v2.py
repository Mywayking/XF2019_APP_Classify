# -*- coding: utf-8 -*-#
"""
@author:Galen
@file: xf_bert_v2.py
@time: 2019-07-31
@description:使用bert提取特征，Bidirectional 训练模型


python3 xf_bert_v2.py

pip install keras
pip install keras_bert

"""
import codecs
import gc
import math
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import CuDNNLSTM, Bidirectional
from keras.layers import Dense, Flatten, SpatialDropout1D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import Sequence
from keras_bert import Tokenizer, load_trained_model_from_checkpoint
from sklearn.preprocessing import LabelBinarizer



data_dir = './data/'
BERT_PRETRAINED_DIR = './chinese_L-12_H-768_A-12/'

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init_op)


def pickle_save(path, data):
    with open(path, 'wb')as f:
        # dumps序列化源数据后写入文件
        f.write(pickle.dumps(data, protocol=4))


def pickle_load(path):
    with open(path, 'rb')as f:
        return pickle.loads(f.read())


def load_data():
    # ============================读入训练集：=======================================
    train = pd.read_csv(data_dir + "apptype_train.dat", header=None, encoding='utf8', delimiter=' ')
    # 以tab键分割，不知道为啥delimiter='\t'会报错，所以先读入再分割。
    train = pd.DataFrame(train[0].apply(lambda x: x.split('\t')).tolist(), columns=['id', 'label', 'comment'])
    print('train', train.shape)
    # =============================读入测试集：======================================
    test = pd.read_csv(data_dir + "app_desc.dat", header=None, encoding='utf8', delimiter=' ')
    test = pd.DataFrame(test[0].apply(lambda x: x.split('\t')).tolist(), columns=['id', 'comment'])
    print('test', test.shape)
    print('load success ')
    # ========================以|为分隔符，把标签分割：===============================
    train['label1'] = train['label'].apply(lambda x: x.split('|')[0])
    train['label2'] = train['label'].apply(lambda x: x.split('|')[1] if '|' in x else 0)  ##第二个标签有些没有，此处补0
    # 去掉样本少于5个的类别
    train = train[~train.label1.isin(['140110', '140805', '140105'])].reset_index(drop=True)

    return train, test


def get_binary_label(data_list):
    lb = LabelBinarizer()
    binary_label = lb.fit_transform(data_list)  # transfer label to binary value
    cate_len = len(lb.classes_)
    print(binary_label.shape)
    # 逆过程
    # yesORno=lb.inverse_transform(p)
    return binary_label, cate_len, lb


def load_bert_model():
    print('load_bert_model')
    # TensorFlow 模型文件，包含预训练模型的权重
    checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
    # 配置文件，记录模型的超参数
    config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
    # 字典文件，记录词条与 id 的映射关系
    dict_file = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
    # 将 BERT 模型载入到 keras
    bert_model = load_trained_model_from_checkpoint(config_file, checkpoint_file, seq_len=64)

    token_dict = {}
    with codecs.open(dict_file, 'r', 'utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    tokenizer = Tokenizer(token_dict)
    return tokenizer, bert_model


class DataGenerator(Sequence):

    def __init__(self, dataX, dataY, batch_size=1, shuffle=True, bert_model=None, tokenizer=None, max_sentence=64):
        self.batch_size = batch_size
        self.dataX = dataX
        self.dataY = dataY
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.max_sentence = max_sentence
        # 验证dataX训练数据和标签是否数量一致
        assert (len(self.dataX) == len(self.dataY))
        self.indexes = np.arange(len(self.dataX))
        self.shuffle = shuffle
        # 打乱数据
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(len(self.dataX) / float(self.batch_size))

    def __getitem__(self, index):
        # 生成每个batch数据
        # 生成batch_size个索引
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 根据索引获取集合中的数据
        batch_X = [self.dataX[k] for k in batch_indexs]
        batch_Y = [self.dataY[k] for k in batch_indexs]

        # 生成数据
        X, y = self.data_generation(batch_X, batch_Y)
        return X, y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_X, batch_Y):
        indices = []
        segments = []
        for i, text in enumerate(batch_X):
            index, segment = self.tokenizer.encode(first=text, max_len=self.max_sentence)
            indices.append(index)
            segments.append(segment)
        word_vec = self.bert_model.predict([np.array(indices), np.array(segments)])
        return word_vec, np.array(batch_Y)


def dl_model(max_sentence, cate_num):
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True), input_shape=(max_sentence, 768)))
    model.add(SpatialDropout1D(0.5))
    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(cate_num, activation='softmax'))
    model.summary()
    return model


def make_callbacks():
    # checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

    #  不再优化时候，调整学习率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', factor=0.8)

    # 当验证集的loss不再下降时，中断训练 patience=2 两个 Epoch 停止
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # all the goodies
    return [reduce_lr, early_stopping]


def bert_data(data_list, bert_model, tokenizer, bert_pkl_path, max_sentence):
    print('bert_data ...')
    if os.path.exists(bert_pkl_path):
        print("loading ", bert_pkl_path)
        return pickle_load(bert_pkl_path)
    indices = []
    segments = []
    for text in tqdm.tqdm(data_list):
        index, segment = tokenizer.encode(first=text, max_len=max_sentence)
        indices.append(index)
        segments.append(segment)
    text_data = [np.array(indices), np.array(segments)]
    del indices
    del segments
    del tokenizer
    gc.collect()
    gc.collect()
    data_vec = bert_model.predict(text_data)
    del bert_model
    del text_data
    gc.collect()
    print("saving ", bert_pkl_path)
    pickle_save(bert_pkl_path, data_vec)
    return data_vec


def train_fit_generator(x_train, y_train, total_epochs, batch_size, steps, max_sentence, cate_len, bert_model,
                        tokenizer):
    data_gen = DataGenerator(x_train, y_train, batch_size, True, bert_model, tokenizer)
    callbacks_list = make_callbacks()
    model = dl_model(max_sentence, cate_len)
    exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1
    lr_init, lr_fin = 0.001, 0.0001
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    optimizer_adam = Adam(lr=0.001, decay=lr_decay)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_adam, metrics=['accuracy'])
    model.fit_generator(data_gen,
                        epochs=total_epochs,
                        callbacks=callbacks_list,
                        # steps_per_epoch=STEPS_PER_EPOCH,
                        # validation_data=val_batch,
                        # validation_steps=VALIDATION_STEPS,
                        verbose=1,  # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
                        )
    return model


def train_fit(x_train, y_train, total_epochs, batch_size, steps, max_sentence, cate_len, bert_model, tokenizer):
    data_vec = bert_data(x_train, bert_model, tokenizer,
                         data_dir + 'train_bert_pickle_{0}.plk'.format(str(max_sentence)), max_sentence)
    callbacks_list = make_callbacks()
    model = dl_model(max_sentence, cate_len)
    exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1
    lr_init, lr_fin = 0.001, 0.0001
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    optimizer_adam = Adam(lr=0.001, decay=lr_decay)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_adam, metrics=['accuracy'])
    model.fit(data_vec, y_train,
              epochs=total_epochs,
              batch_size=batch_size,
              callbacks=callbacks_list,
              validation_split=0.1,
              verbose=1,  # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
              )
    return model


def get_top2(data_np):
    results = pd.DataFrame(data_np)
    first_np = np.zeros(data_np.shape, dtype=int)
    second_np = np.zeros(data_np.shape, dtype=int)
    for j, row in results.iterrows():
        zz = list(np.argsort(row))
        # 第一个标签
        first_np[j, row.index[zz[-1]]] = 1
        # 第二个标签
        second_np[j, row.index[zz[-2]]] = 1
    return first_np, second_np


def predict(model, test_df, batch_size, bert_model, tokenizer, lb, max_sentence):
    # 预测数据
    test_vec = bert_data(test_df['comment'], bert_model, tokenizer,
                         data_dir + 'test_bert_pickle_{0}.plk'.format(str(max_sentence)), max_sentence)
    print('predictions ...')
    predictions = model.predict_proba(test_vec, batch_size=batch_size)
    del test_vec
    gc.collect()
    print('inverse_transform cate ...')
    predictions_1, predictions_2 = get_top2(predictions)
    label1 = lb.inverse_transform(predictions_1)
    label2 = lb.inverse_transform(predictions_2)
    data_save = 'pre_test_{}.csv'.format(datetime.now().strftime('%m%d_%H%M%S'))
    print('saving ...', data_save)
    submission = pd.DataFrame.from_dict({
        'id': test_df['id'],
        'label1': label1,
        'label2': label2,
    })
    submission.to_csv(data_save, sep=',', index=False)


def main():
    epochs = 15
    batch_size = 32
    max_sentence = 64
    train, test = load_data()
    x_train = train['comment']
    y_train, cate_len, lb = get_binary_label(train['label1'])
    tokenizer, bert_model = load_bert_model()
    steps = int(len(x_train) / batch_size) * epochs
    # model = train_fit_generator(x_train, y_train, total_epochs, batch_size,max_sentence, cate_len, bert_model, tokenizer)
    model = train_fit(x_train, y_train, epochs, batch_size, steps, max_sentence, cate_len, bert_model, tokenizer)
    del x_train
    gc.collect()
    predict(model, test, batch_size, bert_model, tokenizer, lb, max_sentence)


main()
