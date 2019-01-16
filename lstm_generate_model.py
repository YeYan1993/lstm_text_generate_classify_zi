import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout,Embedding
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping
from data_get import data_get,train_test_split



def genrate_article(model,word2num,num2word,input_sentences,seq_lenth = 50 ,rounds = 50):
    next_data = []
    for ii in range(rounds):
        # 将所有的输入转成模型输入的形式(X)
        input_data = [word2num[word] for word in input_sentences]
        if len(input_data) >= seq_lenth:
            lenth = len(input_data) - seq_lenth
            input_X = input_data[lenth:]
            input_X = np.array(input_X).reshape((1,-1))
            next_words = model.predict_classes(input_X).tolist()
            next_words = num2word[next_words[0]]
            input_sentences += "" + next_words
            next_data.append(next_words)
        else:
            print("Please input more than 50 words!")
    print(input_sentences)
    return next_words





if __name__ == '__main__':
    """数据处理"""
    all_words,word2num,num2word = data_get()
    X,y = train_test_split(all_words,word2num)

    # 常量值
    seq_lenth = 50
    epoches = 100
    batch_size = 64

    """建立模型"""
    print("Building Model......")
    model = Sequential()
    model.add(Embedding(input_dim=len(word2num)+1,output_dim=128,input_length=seq_lenth))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(len(word2num), activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    # earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    tensorboard = TensorBoard(log_dir="model/log")
    checkpoint_path = "model/best_weights.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, monitor="val_loss", verbose=1)
    model.fit(X, y, batch_size=batch_size, epochs = epoches,callbacks=[cp_callback,tensorboard])
    print(model.summary())

    """预测结果"""
    init ="二十九年了，自从二十九年前他被外门长老唐蓝太爷在襁褓时就捡回唐门时开始，唐门就是他的家，而唐门的暗器就是他的一切。突然，唐三脸色骤然一变，但很快又释然了' \
       '十七道身影，十七道白色的身影，宛如星丸跳跃一般从山腰处朝山顶方向而来，这十七道身影的主人，年纪最小的也超过了五旬，一个个神色凝重，他们身穿的白袍代表的是内门，而"
    next_words = genrate_article(model, word2num,num2word, init, seq_lenth)



