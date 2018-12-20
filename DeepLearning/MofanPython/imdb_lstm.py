#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/10 16:50
# @Author  : Chen Cjv
# @Site    : http://www.cnblogs.com/cjvae/
# @File    : imdb_lstm.py
# @Software: PyCharm
"""
Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
"""
from __future__ import print_function

from keras.preprocessing import sequence
# 最简单的模型类型是Sequential模型 线性层叠
# model = Sequential()
from keras.models import Sequential
# 添加神经层很简单，model.add
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# 构建模型
print('Build model...')
# 最主要的模型，Sequential是一系列网络层按顺序构成的栈
model = Sequential()
# .add 将网络层堆叠起来
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
# 编译模型，必须指明损失函数和优化器
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
# 模型评估，查看指标是否满足要求
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
