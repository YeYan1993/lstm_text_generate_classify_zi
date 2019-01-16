from keras.models import load_model
from lstm_generate_model import genrate_article
import os
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers import Dropout,Embedding
from keras.layers import LSTM
from data_get import data_get


if __name__ == '__main__':
    """预测结果"""
    init = "长老们都没有说话，他们此时还没能从佛怒唐莲的出现中清醒过来。两百年，整整两百年了，佛怒唐莲竟然在一个外门弟子手中出现，" \
           "这意味着什么？这霸绝天下，连唐门自己人也不可能抵挡的绝世暗器代表的绝对是唐门另一个巅峰的来临"
    all_words,word2num,num2word = data_get()
    seq_lenth = 50
    model = Sequential()
    model.add(Embedding(input_dim=len(word2num) + 1, output_dim=128, input_length=seq_lenth))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(len(word2num), activation="softmax"))
    model.load_weights("model/best_weights.hdf5")
    next_words = genrate_article(model, word2num,num2word, init, seq_lenth,rounds=300)
