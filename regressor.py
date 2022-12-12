import pandas as pd
import numpy as np
import time
from regex import R
from xgboost import XGBRegressor, plot_importance, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import math


def predict(pred_year=2016):
    # 数据读入
    train = pd.read_csv('data/{}/processed/train.csv'.format(pred_year))
    test = pd.read_csv('data/{}/processed/test.csv'.format(pred_year))
    attrs = ['author_rank1', 'author_rank2', 'total_citation', 'total_papers', 'h_index']
    if pred_year == 2016:
        attrs += [str(y) for y in range(1992, 2012)]
    elif pred_year == 2022:
        attrs += [str(y) for y in range(1997, 2017)]
    else:
        raise NotImplementedError

    x_train = train[attrs]
    y_train = train['result']
    x_test = test[attrs]
    y_test = test['result']
    print("All the data has been read")

    # Pipeline设置：归一化与降维
    # pipe_lr = Pipeline([('sc',StandardScaler()),
    #                     ('pca', PCA(n_components=2)),
    #                     ('svr', SVR())])
    # t1 = time.perf_counter()
    # pipe_lr.fit(x_train, y_train)
    # t2 = time.perf_counter()
    # print("fit time:", t2 - t1)
    # y = pipe_lr.predict(x_test)
    # t3 = time.perf_counter()
    # print("Predict time:", t3 - t2)
    # rmse = math.sqrt(mean_squared_error(y_test, y))
    # print(rmse)

    # 网格搜索
    # regressor = XGBRegressor(eval_metric='rmse', n_jobs=-1)
    # param_grid = {'max_depth': [x for x in range(3,11)],
    #                 'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    #                 'n_estimators' : [20 * x for x in range(1, 11)]}
    # grid_search = GridSearchCV(regressor, param_grid, scoring='neg_mean_squared_error')
    # grid_result = grid_search.fit(x_train, y_train)
    # print("Best: %f using %s" % (grid_result.best_score_,grid_search.best_params_))

    # 模型选择
    # DecisionTreeRegressor(max_depth=7)
    # BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=7), n_estimators=40)
    # AdaboostRegressor(base_estimator=DecisionTreeRegressor(max_depth=7), learning_rate=0.1, n_estimators=150)
    # GradientBoostingRegressor(learning_rate=0.1, n_estimators=100)
    # XGBRegressor(max_depth=7, learning_rate=0.1, n_estimators=50, eval_metric='rmse')
    models = ['LinearRegression', 'DecisionTreeRegressor',
         'RandomForestRegressor', 'AdaBoostRegressor', 'GradientBoostingRegressor',
         'BaggingRegressor']
    for model in models:
        print("===============================")
        print(model)
        regressor = eval(model)()  # 可设置模型具体参数
        t1 = time.perf_counter()
        regressor.fit(x_train, y_train)
        t2 = time.perf_counter()
        print("fit time:", t2 - t1)
        y = regressor.predict(x_test)
        t3 = time.perf_counter()
        print("Predict time:", t3 - t2)
        rmse = math.sqrt(mean_squared_error(y_test, y))
        print("RMSE: ", rmse)

    # 特征重要性评估
    # l = ['author_rank1', 'author_rank2', 'total_citation', 'total_papers', 'h_index',
    #              '1992', '1993', '1994', '1995', '1996',
    #              '1997', '1998', '1999', '2000', '2001',
    #              '2002', '2003', '2004', '2005', '2006',
    #              '2007', '2008', '2009', '2010', '2011']
    # print([l[x] for x in np.argsort(-regressor.feature_importances_)])

    # 保存数据
    # with open("output/gbrt_output.txt", 'w', encoding='utf8') as fo:
    #     with open("data/{}/raw/citation_test.txt".format(pred_year), 'r', encoding='utf8') as ft:
    #         lines = ft.readlines()
    #         for i in range(len(lines)):
    #             words = lines[i].strip().split('\t')
    #             _id = words[0]
    #             fo.write(_id + '\t' + str(y[i]) + '\n')


if __name__ == '__main__':
    predict(pred_year=2022)
