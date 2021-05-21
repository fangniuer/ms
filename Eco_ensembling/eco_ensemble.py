# -*- coding: utf-8 -*-
# @Time    : 2020/10/31 15:31
# @Author  : Wujun Dai
# @Email   : wjdainefu@126.com
# @File    : eco_ensemble.py
# @Software: PyCharm

import pandas as pd
import numpy as np

from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import *
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def eco_ensemble(df, df_his, eco_model_set, ensemble_method_list, top_n,
                 ensemble_seed=123, ensemble_predict=False, ):

    ensemble_model_list = []
    for ensemble_method in ensemble_method_list:
        if ensemble_method == 'AdaBoostRegressor':
            ensembled = AdaBoostRegressor(random_state=ensemble_seed, n_estimators=100)
            ensemble_model_list.append(ensembled)

        elif ensemble_method == 'BaggingRegressor':
            ensembled = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=ensemble_seed)
            ensemble_model_list.append(ensembled)

        elif ensemble_method == 'ExtraTreesRegressor':
            ensembled = ExtraTreesRegressor(n_estimators=100, random_state=ensemble_seed)
            ensemble_model_list.append(ensembled)

        elif ensemble_method == 'GradientBoostingRegressor':
            ensembled = GradientBoostingRegressor(random_state=ensemble_seed)
            ensemble_model_list.append(ensembled)

        elif ensemble_method == 'RandomForestRegressor':
            ensembled = RandomForestRegressor(max_depth=2, random_state=ensemble_seed)
            ensemble_model_list.append(ensembled)

        elif ensemble_method == 'StackingRegressor':
            estimators = [('lr', RidgeCV()), ('svr', LinearSVR(random_state=42))]
            ensembled = StackingRegressor(
                estimators=estimators,
                final_estimator=RandomForestRegressor(n_estimators=10, random_state=42))
            ensemble_model_list.append(ensembled)

        elif ensemble_method == 'VotingRegressor':
            reg1 = GradientBoostingRegressor(random_state=1)
            reg2 = RandomForestRegressor(random_state=1)
            reg3 = LinearRegression()
            ensembled = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
            ensemble_model_list.append(ensembled)

        elif ensemble_method == 'HistGradientBoostingRegressor':
            ensembled = HistGradientBoostingRegressor()
            ensemble_model_list.append(ensembled)

    eco_pre = df[df.model.isin(eco_model_set)]
    ensemble_df = eco_pre.pivot_table(index=['year', 'label', 'date'],
                                      columns='model', values='prediction').reset_index()
    train_df, test_df = ensemble_df.query('label=="train"'), ensemble_df.query('label=="test"')
    y_train, y_test = train_df.date, test_df.date
    # x_train, x_test = train_df[eco_model_set], test_df[eco_model_set]

    eco_train_rmses = []
    eco_test_rmses = []
    for model_str in eco_model_set:
        eco_pre_test = ensemble_df.query('label=="test"')[model_str]
        eco_test_rmse = round(mean_squared_error(y_test, eco_pre_test, squared=False), 2)
        eco_pre_train = ensemble_df.query('label=="train"')[model_str]
        eco_train_rmse = round(mean_squared_error(y_train, eco_pre_train, squared=False), 2)
        eco_test_rmses.append(eco_test_rmse)
        eco_train_rmses.append(eco_train_rmse)

    eco_model_rmse = pd.DataFrame({'model': eco_model_set, 'train': eco_train_rmses, 'test': eco_test_rmses})
    top_model_set = eco_model_rmse.nsmallest(top_n, 'test', keep='first').model.to_list()

    model_list = eco_model_set + ensemble_method_list
    eco_pre = df[df.model.isin(top_model_set)]
    ensemble_df = eco_pre.pivot_table(index=['year', 'label', 'date'],
                                      columns='model', values='prediction').reset_index()
    train_df, test_df = ensemble_df.query('label=="train"'), ensemble_df.query('label=="test"')
    y_train, y_test = train_df.date, test_df.date
    x_train, x_test = train_df[top_model_set], test_df[top_model_set]

    """
    Ensemble methods
    The goal is to predict numerical value, so the classification ensemble model is not included
    """
    for ensemble_model in ensemble_model_list:
        est = ensemble_model.fit(x_train, y_train)

        y_pre_train = np.around(est.predict(x_train))
        train_rmse = round(mean_squared_error(y_train, y_pre_train, squared=False), 2)

        y_pre_test = np.around(est.predict(x_test))
        test_rmse = round(mean_squared_error(y_test, y_pre_test, squared=False), 2)
        # test_df['ensemble'] = y_pre
        eco_train_rmses.append(train_rmse)
        eco_test_rmses.append(test_rmse)
    result = pd.DataFrame({'model': model_list, 'train_rmse': eco_train_rmses, 'test_rmse': eco_test_rmses})

    if ensemble_predict:
        phenophase, species = df.phenophase.values[0], df.species.values[0]
        species_his = df_his.query('phenophase==@phenophase & species==@species')
        eco_his = species_his[species_his.model.isin(top_model_set)]
        eco_his['label'] = eco_his.date.apply(lambda x: 'ob' if pd.notna(x) else 'pre')
        # The 'date' column contains Nan, if use pivot directly, the row containing the missing value will be eliminated
        train_df = eco_his.pivot_table(index=['year', 'label', 'date'], columns='model', values='y_pred').reset_index()
        his_pre = eco_his.pivot_table(index=['year', 'label'], columns='model', values='y_pred').reset_index()
        y_train, x_train = train_df.date, train_df[top_model_set]
        x_pre = his_pre[top_model_set]

        # Ensemble prediction: selected "RandomForestRegressor" and  "VotingRegressor" for ensemble prediction
        ensemble_selected = [ensemble_model_list[0], ensemble_model_list[5]]
        eco_pre_his = species_his.pivot_table(index=['year'], columns='model', values='y_pred').reset_index()
        for ensemble_model in ensemble_selected:
            est = ensemble_model.fit(x_train, y_train)
            y_pre = np.around(est.predict(x_pre))
            model_str = str(ensemble_model).split("(")[0]
            eco_pre_his[model_str] = y_pre

        ensemble_df = eco_pre_his.merge(train_df[['year', 'date']], 'left')
        ensemble_df.rename(columns={'date': 'OB'}, inplace=True)
        return ensemble_df
    else:
        return result


if __name__ == "__main__":

    # test
    # eco_train_test_prediction = pd.read_csv('./pre/prediction_obs.csv')\
    #     .query('phenophase=="LUD" and species=="山槐" & algorithm=="dual_annealing"')
    # eco_history_prediction = pd.read_csv('./pre/prediction_his.csv').\
    #     query('phenophase=="LUD" and species=="山杏" & algorithm=="dual_annealing"')


    top_n = 2
    eco_model_set = ['PU', 'DH', 'AL', 'GU', 'GDD', 'RI', 'M1', 'PT'] # 3
    eco_train_test_prediction = pd.read_csv('./pre/prediction_obs.csv')
    eco_his = pd.read_csv('./pre/prediction_his.csv')

    # eco_train_test_prediction = pd.read_csv('./pre/prediction_obs.csv').query('phenophase=="LUD"')
    # eco_his = pd.read_csv('./pre/prediction_his.csv').query('phenophase=="LUD"')

    # ensemble_method_list = ['RandomForestRegressor', "ExtraTreesRegressor"]
    ensemble_method_list = ["RandomForestRegressor", "ExtraTreesRegressor", "BaggingRegressor",
                            "GradientBoostingRegressor", "AdaBoostRegressor", "VotingRegressor",
                            "StackingRegressor", "HistGradientBoostingRegressor"]


    # eco_model_set = ['GDD', 'DH', 'AL', 'PU'] # 4
    #  ['TC', 'RI', 'AMC', 'GDD', 'EX', 'UF', 'FU', 'PU', 'DH', 'PT', 'M1',
    #  'SE', 'UC', 'AL', 'UM', 'PA', 'GU', 'DOR', 'DR', 'FP']

    # ensemble_evaluate_result = eco_train_test_prediction.groupby(['phenophase', 'species']).\
    #     apply(lambda x: eco_ensemble(x, eco_model_set, ensemble_method_list, top_n=top_n))

    ensemble_metric_top = eco_train_test_prediction.groupby(['phenophase', 'species']).\
        apply(lambda x: eco_ensemble(x, eco_his, eco_model_set, ensemble_method_list, top_n))
    ensemble_metric_top.to_csv(f'./pre/ensemble_metric_top{top_n}.csv', encoding='utf_8_sig')

    ensemble_prediction = eco_train_test_prediction.groupby(['phenophase', 'species']). \
        apply(lambda x: eco_ensemble(x, eco_his, eco_model_set, ensemble_method_list, top_n, ensemble_predict=True))
    ensemble_prediction.to_csv(f'./pre/ensemble_prediction{top_n}.csv', encoding='utf_8_sig')

    # print(ensemble_train_result)
    # print(prediction)
