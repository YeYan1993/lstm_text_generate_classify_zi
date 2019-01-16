import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout,Embedding
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping
import collections
from keras.utils import to_categorical

class LSTM_model():
    def __init__(self):
        self.steps = 3
        self.seq_lenth = 50
        self.epoches = 100
        self.batch_size = 64
        self.checkpoint_path = "model/best_weights.hdf5"
        self.path = "data/斗罗大陆.txt"
        self.all_words,self.word2num, self.num2word = self.data_get()
        self.X, self.y = self.data_train_test_split()



    def data_get(self):
        data = open(self.path, encoding='utf-8').read()
        data = data[0:(len(data) // 500)]
        print(len(data))
        data = data.replace('\n', '')
        data = data.replace('-', '')
        data = data.replace(' ', '')
        self.all_words = [word for word in data]
        counter = collections.Counter(self.all_words)
        counter_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        words, _ = zip(*counter_sorted)
        self.word2num = dict(zip(words, range(len(words))))
        self.num2word = dict(zip(range(len(words)), words))
        return self.all_words, self.word2num, self.num2word

    def data_train_test_split(self):
        """构建训练集和测试集"""
        self.X = []
        self.y = []
        for i in range(0, len(self.all_words) - self.seq_lenth, self.steps):
            train_seq = self.all_words[i:i + self.seq_lenth]
            predic_seq = self.all_words[i + self.seq_lenth]
            self.X.append(np.array([self.word2num[train_seq_single] for train_seq_single in train_seq]))
            self.y.append(np.array(self.word2num[predic_seq]))
        self.X = np.array(self.X)
        self.y = to_categorical(self.y, num_classes=len(self.word2num))
        return self.X, self.y

    def model_build(self):
        print("Building Model......")
        self.model = Sequential()
        self.model.add(Embedding(input_dim=len(self.word2num) + 1, output_dim=128, input_length=self.seq_lenth))
        self.model.add(LSTM(256))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(len(self.word2num), activation="softmax"))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        # earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        tensorboard = TensorBoard(log_dir="model/log")
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        cp_callback = ModelCheckpoint(self.checkpoint_path, save_weights_only=True, monitor="val_loss", verbose=1)
        self.model.fit(self.X, self.y, batch_size=self.batch_size, epochs=self.epoches, callbacks=[cp_callback, tensorboard])
        print(self.model.summary())

class text_predict_articles():
    def __init__(self):
        self.lstm_model_weights_path = "model/best_weights.hdf5"
        self.seq_lenth = 50
        self.rounds = 100
        self.lstm_model = LSTM_model()
        self.all_words, self.word2num, self.num2word = self.lstm_model.data_get()
        self.model = Sequential()
        self.model.add(Embedding(input_dim=len(self.word2num) + 1, output_dim=128, input_length=self.seq_lenth))
        self.model.add(LSTM(256))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(len(self.word2num), activation="softmax"))
        self.model.load_weights(self.lstm_model_weights_path)




    def genrate_article(self, input_sentences):
        next_data = []
        for ii in range(self.rounds):
            # 将所有的输入转成模型输入的形式(X)
            input_data = [self.word2num[word] for word in input_sentences]
            if len(input_data) >= self.seq_lenth:
                lenth = len(input_data) - self.seq_lenth
                input_X = input_data[lenth:]
                input_X = np.array(input_X).reshape((1, -1))
                next_words = self.model.predict_classes(input_X).tolist()
                next_words = self.num2word[next_words[0]]
                input_sentences += "" + next_words
                next_data.append(next_words)
            else:
                print("Please input more than 50 words!")
        print(input_sentences)
        return next_words

if __name__ == '__main__':
    text_articles = text_predict_articles()
    input_sentences = "长老们都没有说话，他们此时还没能从佛怒唐莲的出现中清醒过来。两百年，整整两百年了，佛怒唐莲竟然在一个外门弟子手中出现，" \
           "这意味着什么？这霸绝天下，连唐门自己人也不可能抵挡的绝世暗器代表的绝对是唐门另一个巅峰的来临"
    generated_articles = text_articles.genrate_article(input_sentences)
    print(generated_articles)