import os
import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
import tqdm
import math


def predict(pred_year=2016):
    print("Auto Arima for 30 years.")
    timecnt = pd.read_csv('data/{}/processed/cnt_test_30_new.csv'.format(pred_year))
    test = pd.read_csv('data/{}/processed/test.csv'.format(pred_year))
    timelist = [str(i) for i in range(1982, 2012)]
    x_test = timecnt[timelist]
    y_test = np.array(test['result'])
    y = np.array(test['total_citation'])
    for i in tqdm.trange(x_test.shape[0]):
        train = np.array(x_test.iloc[i])
        if np.sum(train) != 0:
            model = auto_arima(train, start_p=0, start_q=0, max_p=6, max_q=6, max_d=2,
                               seasonal=False, njob=-1, error_action="ignore")
            model.fit(train)
            forecast = model.predict(n_periods=5)
            for cnt in forecast:
                y[i] += cnt
    print("Prediction completed.")
    os.makedirs("output/{}".format(pred_year), exist_ok=True)
    with open("output/{}/arima_10.txt".format(pred_year), 'w', encoding='utf8') as fo:
        with open("data/{}/raw/citation_test.txt".format(pred_year), 'r', encoding='utf8') as ft:
            lines = ft.readlines()
            for i in range(len(lines)):
                words = lines[i].strip().split('\t')
                _id = words[0]
                fo.write(_id + '\t' + str(y[i]) + '\n')
    rmse = math.sqrt(mean_squared_error(y_test, y))
    print(rmse)


if __name__ == '__main__':
    predict(pred_year=2016)
