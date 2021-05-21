# -*- coding: utf-8 -*-
# @Time    : 2020/10/24 10:31
# @Author  : Wujun Dai
# @Email   : wjdainefu@126.com
# @File    : eco_models_run.py
# @Software: PyCharm

from eco_models import *
from pyswarms.backend.topology import Pyramid

import datetime
import numpy as np
import os
import pandas as pd

if __name__ == '__main__':

    """
    The custom parameters
    """
    lud = pd.read_csv('./data/phe_cleaned.csv').query("phenophase=='LUD'")
    phenophases_set = ['LUD']  # ['开花始期', '开始展叶期']
    # models_set = ['TC', 'RI', 'AMC', 'GDD', 'EX', 'UF', 'FU', 'PU', 'DH', 'PT', 'M1',
    #                                  'SE', 'UC', 'AL', 'UM', 'PA', 'GU', 'DOR', 'DR', 'FP']
    models_set = ['GDD', 'DH', 'AL']  # ['TC', 'RI']
    # result_dir = './results'
    result_dir = './pre'
    algorithms_set = [
        'dual_annealing']  # ['dual_annealing', 'differential_evolution', 'PSO', 'CMAES', 'bayesian_opt']
    # metric_results_file_path = result_dir + '/' + 'metric.csv'
    # predictions_obs_file_path = result_dir + '/' + 'prediction_obs.csv'
    # train_test_tuning_file_path = result_dir + '/' + 'train_test_tuning.csv'
    # species_set = sorted(list(set(lud['species']) - {'东北连翘', '山杏', '紫丁香', '金银忍冬'}))[8:10]
    species_set = sorted(list(set(lud['species'])))
    # species_set = ['东北山梅花', '金银忍冬']  # ['紫丁香', '山杏', '东北山梅花', '金银忍冬']
    metric_results_file_path = result_dir + '/' + 'metric.csv'
    predictions_obs_file_path = result_dir + '/' + 'prediction_obs.csv'
    save_train_tuning_history_bool = True
    train_test_tuning_file_path = result_dir + '/' + 'train_test_tuning.csv'
    predict_history_bool = True
    predictions_his_file_path = result_dir + '/' + 'prediction_his.csv'
    metric_fit_file_path = result_dir + '/' + 'metric_fit.csv'

    algorithm_parameters = {'max_iter': 1000,
                            'algorithm_RandomSeed': 3,
                            'PSO_Options': {'c1': 0.5, 'c2': 0.3, 'w': 0.9},
                            'PSO_Topology': Pyramid(static=False),
                            'bayesian_opt_IterRatio': 0.2,  # The product of 'max_iter' and it must be larger than 1.
                            'split_seed': 303,
                            'KFold_nSplits': 3
                            }

    weather_data_file_path = './data/weather_harbin.csv'
    flower_data_file_path = './data/phe_cleaned.csv'
    models_parameters_file_path = './data/parameter_range.range'


    """
    The codes below do not need to modify
    """
    metric_results = pd.DataFrame()
    predictions_obs = pd.DataFrame()
    train_tuning_history_df = pd.DataFrame()
    predictions_his = pd.DataFrame()
    metric_fit = pd.DataFrame()
    for phenophase_name in phenophases_set:
        phenophases_file_path = result_dir + '/' + phenophase_name + '/'
        for model_name in models_set:

            for algorithm_name in algorithms_set:
                species_predictions_df = pd.DataFrame()
                results_file_path = phenophases_file_path + algorithm_name + '/'
                log_file_path = results_file_path + model_name + '.log'

                if os.path.isdir(results_file_path):
                    pass
                else:
                    os.makedirs(results_file_path)

                for species_name in species_set:
                    start_time = datetime.datetime.now()
                    model = EcoModels(model_name, algorithm_name, algorithm_parameters, species_name, phenophase_name)
                    model.load_data(weather_data_file_path, flower_data_file_path)
                    model.load_models_parameters(models_parameters_file_path)
                    time = datetime.datetime.now()
                    model.train_and_test()

                    if save_train_tuning_history_bool:
                        train_tuning_history = pd.concat([pd.DataFrame({'phenophase': phenophase_name,
                                                          'species': species_name,
                                                          'model': model_name,
                                                          'algorithm': algorithm_name,
                                                          'seed': algorithm_parameters['split_seed']},
                                                          index=model.train_tuning_history.index),
                                                          model.train_tuning_history], axis=1)

                        train_tuning_history_df = train_tuning_history_df.append(train_tuning_history)

                    else:
                        elapsed_time_train_test = datetime.datetime.now()

                    end_time = datetime.datetime.now()

                    log_file_handle = open(log_file_path, mode='a+')
                    print('Species: {}, Start time: {}'.format(species_name, start_time.strftime('%Y-%m-%d %H:%M:%S')),
                          file=log_file_handle)
                    print('', file=log_file_handle)
                    print('Cross validation:', time.strftime('%Y-%m-%d %H:%M:%S'),
                          file=log_file_handle)
                    print('The best parameters of cross validation:', model.train_best_parameters, file=log_file_handle)
                    print('The RMSE of training process in cross validation:', np.round(model.train_RMSE, 3).tolist(),
                          file=log_file_handle)
                    print('The RMSE of testing process in cross validation:', np.round(model.test_RMSE, 3).tolist(),
                          file=log_file_handle)
                    print('The AIC of training process in cross validation:', np.round(model.train_AIC, 3).tolist(),
                          file=log_file_handle)
                    print('The AIC of testing process in cross validation:', np.round(model.test_AIC, 3).tolist(),
                          file=log_file_handle)
                    print('The NSE of training process in cross validation:', np.round(model.train_NSE, 3).tolist(),
                          file=log_file_handle)
                    print('The NSE of testing process in cross validation:', np.round(model.test_NSE, 3).tolist(),
                          file=log_file_handle)
                    print('', file=log_file_handle)
                    print('', file=log_file_handle)
                    print('End time:', end_time.strftime('%Y-%m-%d %H:%M:%S'), file=log_file_handle)
                    print('\n', file=log_file_handle)
                    log_file_handle.close()


                    metric_results = metric_results.append(pd.DataFrame({'phenophase': phenophase_name,
                                                   'species': species_name,
                                                   'model': model_name,
                                                   'algorithm': algorithm_name,
                                                   'elapsed': end_time - start_time,
                                                   'train_rmse': model.train_RMSE,
                                                   'test_rmse': model.test_RMSE,
                                                   'train_AIC': model.train_AIC,
                                                   'test_AIC': model.test_AIC,
                                                   'train_NSE': model.train_NSE,
                                                   'test_NSE': model.test_NSE,
                                                   'best_params': [model.train_best_parameters],
                                                   'seed': algorithm_parameters['split_seed']
                                                   }, index=[0]))

                    prediction_train_test = model.predictions_df
                    train_test_predictions = pd.concat([pd.DataFrame({'phenophase': phenophase_name,
                                                                      'model': model_name,
                                                                      'algorithm': algorithm_name,
                                                                      'seed': algorithm_parameters['split_seed'],
                                                                      'iters': algorithm_parameters['max_iter']},
                                                                    index=prediction_train_test.index),
                                                        prediction_train_test], axis=1)
                    predictions_obs = predictions_obs.append(train_test_predictions)

                    if predict_history_bool:
                        model.fit()
                        metric_fit = metric_fit.append(pd.DataFrame({'phenophase': phenophase_name,
                                                                     'species': species_name,
                                                                     'model': model_name,
                                                                     'algorithm': algorithm_name,
                                                                     'fit_rmse': model.fit_RMSE,
                                                                     'fit_AIC': model.fit_AIC,
                                                                     'fit_NSE': model.fit_NSE,
                                                                     'best_params': [model.fit_best_parameters],
                                                                     'seed': algorithm_parameters['split_seed']
                                                                     }, index=[0]))

                        model.predict()
                        model.predictions_to_DataFrame()
                        prediction_fit = model.predictions_df
                        history_predictions = pd.concat([pd.DataFrame({'phenophase': phenophase_name,
                                                                          'model': model_name,
                                                                          'algorithm': algorithm_name,
                                                                          'seed': algorithm_parameters['split_seed'],
                                                                          'iters': algorithm_parameters['max_iter']},
                                                                         index=prediction_fit.index),
                                                            prediction_fit], axis=1)
                        predictions_his = predictions_his.append(history_predictions)


    if os.path.exists(metric_results_file_path):
        metric_results = pd.read_csv(metric_results_file_path).append(metric_results)
    metric_results.to_csv(metric_results_file_path, encoding='utf_8_sig', index=False)

    if os.path.exists(predictions_obs_file_path):
        predictions_obs = pd.read_csv(predictions_obs_file_path).append(predictions_obs)
    predictions_obs.to_csv(predictions_obs_file_path, index=False, encoding='utf_8_sig')

    if save_train_tuning_history_bool:
        if os.path.exists(train_test_tuning_file_path):
            train_tuning_history_df = pd.read_csv(train_test_tuning_file_path).append(train_tuning_history_df)
        train_tuning_history_df.to_csv(train_test_tuning_file_path, index=False, encoding='utf_8_sig')

    if predict_history_bool:
        if os.path.exists(predictions_his_file_path):
            predictions_his = pd.read_csv(predictions_his_file_path).append(predictions_his)
        predictions_his.to_csv(predictions_his_file_path, index=False, encoding='utf_8_sig')

        if os.path.exists(metric_fit_file_path):
            metric_fit = pd.read_csv(metric_fit_file_path).append(metric_fit)
        metric_fit.to_csv(metric_fit_file_path, index=False, encoding='utf_8_sig')