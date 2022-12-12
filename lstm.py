import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import mean_squared_error
import math
import os


def predict(pred_year=2016, dim=2):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)

    # train = pd.read_csv('../dataset/train.csv')
    # test = pd.read_csv('../dataset/test.csv')
    # x_train = np.array(train[['1992', '1993', '1994', '1995', '1996',
    #                  '1997', '1998', '1999', '2000', '2001',
    #                  '2002', '2003', '2004', '2005', '2006',
    #                  '2007', '2008', '2009', '2010', '2011']])
    # y_train = np.array(train['result'])
    # x_test = np.array(test[['1992', '1993', '1994', '1995', '1996',
    #                '1997', '1998', '1999', '2000', '2001',
    #                '2002', '2003', '2004', '2005', '2006',
    #                '2007', '2008', '2009', '2010', '2011']])
    # y_test = np.array(test['result'])

    x_train = np.load("data/{}/processed/train_feature_seq.npy".format(pred_year))
    x_test = np.load("data/{}/processed/test_feature_seq.npy".format(pred_year))
    y_train = np.load("data/{}/processed/y_train.npy".format(pred_year))
    y_test = np.load("data/{}/processed/y_test.npy".format(pred_year))
    n_seq = x_train.shape[1]

    x_train, y_train = shuffle(x_train, y_train)
    sc = StandardScaler()
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    x_train = x_train.reshape(x_train.shape[0], n_seq, 2)
    x_test = x_test.reshape(x_test.shape[0], n_seq, 2)

    if dim == 1:
        x_train = x_train[:, :, 0: 1]
        x_test = x_test[:, :, 0: 1]

    print("All the data has been read")

    model = Sequential()
    # model.add(LSTM(20, recurrent_activation='sigmoid', input_shape=(20, 1), dropout=0.2))
    if dim == 2:
        model.add(Bidirectional(LSTM(20, recurrent_activation='sigmoid', dropout=0.2), input_shape=(20, 2)))   
    elif dim == 1:
        model.add(Bidirectional(LSTM(20, recurrent_activation='sigmoid', dropout=0.2), input_shape=(20, 1)))
    else:
        raise NotImplementedError
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='mean_squared_error')
    print(model.summary())
    history = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.2, verbose=1)
    y = model.predict(x_test, verbose=1)

    # with open("output/VanillaLSTM.txt", 'w', encoding='utf8') as fo:
    #     with open("data/{}/raw/citation_test.txt".format(pred_year), 'r', encoding='utf8') as ft:
    #         lines = ft.readlines()
    #         for i in range(len(lines)):
    #             words = lines[i].strip().split('\t')
    #             _id = words[0]
    #             fo.write(_id + '\t' + str(y[i, 0]) + '\n')
    rmse = math.sqrt(mean_squared_error(y_test.ravel(), y.ravel()))
    print("dim", dim, rmse)


if __name__ == '__main__':
    predict(pred_year=2022, dim=2)
