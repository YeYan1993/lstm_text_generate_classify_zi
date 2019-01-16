import numpy as np
from keras.utils import to_categorical
import collections

def data_get():
    data = open("data/斗罗大陆.txt", encoding='utf-8').read()
    data = data[0:(len(data) // 500)]
    print(len(data))
    data = data.replace('\n', '')
    data = data.replace('-', '')
    data = data.replace(' ', '')
    all_words = [word for word in data]
    counter = collections.Counter(all_words)
    counter_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    words, _ = zip(*counter_sorted)
    word2num = dict(zip(words, range(len(words))))
    num2word = dict(zip(range(len(words)),words))
    return all_words,word2num,num2word

def train_test_split(all_words,word2num):
    """构建训练集和测试集"""
    seq_lenth = 50
    steps = 3
    X = []
    y = []
    for i in range(0, len(all_words) - seq_lenth, steps):
        train_seq = all_words[i:i + seq_lenth]
        predic_seq = all_words[i + seq_lenth]
        X.append(np.array([word2num[train_seq_single] for train_seq_single in train_seq]))
        y.append(np.array(word2num[predic_seq]))
    X = np.array(X)
    y = to_categorical(y, num_classes=len(word2num))
    return X,y






