# -*- coding: utf-8 -*-
# @Time    : 2020/10/24 10:33
# @Author  : Wujun Dai
# @Email   : wjdainefu@126.com
# @File    : eco_models.py
# @Software: PyCharm

# !/usr/bin/env python
# -*- coding: utf-8 -*-


from bayes_opt import BayesianOptimization
from models_functions import *
from optimize_algorithms import *
# from pyswarms.backend.topology import Pyramid
from scipy.optimize import dual_annealing, differential_evolution
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

# import datetime
import numpy as np
# import os
import pandas as pd
import pyswarms as ps
import re

"""
The following definition is the class containing the framework of Eco models.
"""


class EcoModels(object):

    def __init__(self, model_name, algorithm_name, algorithm_parameters, species_name, phenophase_name):
        super(EcoModels, self).__init__()

        # The names of Eco models
        self.supported_models = ['TC', 'RI', 'AMC', 'GDD', 'EX', 'UF', 'FU', 'PU', 'DH', 'PT', 'M1',
                                 'SE', 'UC', 'AL', 'UM', 'PA', 'GU', 'DOR', 'DR', 'FP']

        # The name of Eco model
        self.model_name = model_name

        # The dictionary storing parameters used by Eco model
        self.models_parameters = {}  # {
        #  model_name: {'keys': ['x', ...],
        #               'init': [1, ...],
        #               'min':  [0, ...],
        #               'max':  [2, ...]
        #               },
        #  ...
        #  }

        # The names of optimization algorithms
        self.supported_algorithms = ['dual_annealing', 'differential_evolution', 'PSO', 'CMAES', 'bayesian_opt']

        # The name of optimization algorithm
        self.algorithm_name = algorithm_name  # ['dual_annealing', 'differential_evolution', 'PSO', 'CMAES', 'bayesian_opt']

        # The dictionary storing parameters used by algorithms
        self.algorithm_parameters = algorithm_parameters  # {'max_iter': 100,
        #  'algorithm_RandomSeed': 3,
        #  'PSO_Options': {'c1': 0.5, 'c2': 0.3, 'w':0.9},
        #  'PSO_Topology': Pyramid(static = False),
        #  'bayesian_opt_IterRatio': 0.1,
        #  'split_seed': 303,
        #  'KFold_nSplits': 5
        #  }

        # The name of plant
        self.species_name = species_name

        # The name of phenophase
        self.phenophase_name = phenophase_name

        # The dictionary stroing constants used by Eco models
        self.constants = {
            'weather': pd.core.frame.DataFrame,  # 天气数据
            'weather_years': set or list,  # 天气数据包含的年份
            'T_mean_column': 'mean_TEM',  # 天气数据中的温度均值列名
            'T_min_column': 'min_TEM',  # 天气数据中的温度最小值列名
            'T_max_column': 'max_TEM',  # 天气数据中的温度最大值列名
            'L_column': 'PHO',  # 天气数据中的昼长列名
            'flower': pd.core.frame.DataFrame,  # 花期数据
            'flower_name': list,  # 花期数据中的物种名
            'flower_years': set or list,  # 花期数据中包含的年份
            'flower_date_column': 'date',  # 花期数据中的日期列名
            'weather_flower_years': list,  # 天气和花期数据共同包含的年份
            'year_column': 'year',  # 天气和花期数据中的年份列名
            'max_error': 365,  # 当设置的参数不能满足序列中的Sf>C或Sc>F时，指定一个较大的误差
            'max_date': 365,  # 当设置的参数不能满足序列中的Sf>C或Sc>F时，指定一个较大的预测值
            'last_year_start_date_neg': -181,
            'parameters_to_be_sorted': [['T_l', 'T_o', 'T_u'],  # 在'base functions'中部分参数存在大小限制关系，而约束条件中无法体现，
                                        ['C_dr', 'C_crit'],  # 因此需要对'base functions'中的此类参数及其优化后的结果进行排序处理,
                                        ['C_pr', 'C_crit']]  # 子列表中的参数按照限制关系由小到大排列
        }

        # The dictionary storing variables used by Eco models
        self.variables = {
            'years': list or np.ndarray  # 训练、验证、构建最终模型等阶段所用数据对应的年份
        }

        # Check the input
        if self.model_name in self.supported_models:
            pass
        else:
            raise ValueError(
                "Unsupported input of [model_name] for initializing the class 'EcoModels': '" + self.model_name + "'. [model_name] should be in " + str(
                    self.supported_models))

        if self.algorithm_name in self.supported_algorithms:
            pass
        else:
            raise ValueError(
                "Unsupported input of [algorithm] for initializing the class 'EcoModels': '" + self.algorithm_name + "'. [algorithm] should be in " + str(
                    self.supported_algorithms))

    def construct_model(self, parameters, indicator):
        try:
            parameter_key = 't_s'
            parameters_keys = self.models_parameters[self.model_name]['keys']
            parameters_mins = self.models_parameters[self.model_name]['min']
            self.constants['last_year_start_date_neg'] = parameters_mins[parameters_keys.index(parameter_key)]
        except:
            raise ValueError(
                "The parameter '" + parameter_key + "' could not be found for the model '" + self.model_name + "' in the row named 'params' in models' parameters file!")

        if self.model_name == 'TC':
            return TC(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], indicator
            )
        elif self.model_name == 'RI':
            return RI(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], indicator
            )
        elif self.model_name == 'AMC':
            return AMC(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], indicator
            )
        elif self.model_name == 'GDD':
            return GDD(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], indicator
            )
        elif self.model_name == 'EX':
            return EX(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], indicator
            )
        elif self.model_name == 'UF':
            return UF(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], indicator
            )
        elif self.model_name == 'FU':
            return FU(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], indicator
            )
        elif self.model_name == 'PU':
            return PU(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['L_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], indicator
            )
        elif self.model_name == 'DH':
            return DH(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_min_column'], self.constants['T_max_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], indicator
            )
        elif self.model_name == 'PT':
            return PT(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['T_min_column'], self.constants['T_max_column'],
                self.constants['L_column'],
                self.constants['flower_date_column'], self.constants['max_error'], self.constants['max_date'], indicator
            )
        elif self.model_name == 'M1':
            return M1(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['L_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], indicator
            )
        elif self.model_name == 'SE':
            return SE(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], self.constants['last_year_start_date_neg'],
                indicator
            )
        elif self.model_name == 'UC':
            return UC(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], self.constants['last_year_start_date_neg'],
                indicator
            )
        elif self.model_name == 'AL':
            return AL(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], self.constants['last_year_start_date_neg'],
                indicator
            )
        elif self.model_name == 'UM':
            return UM(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], self.constants['last_year_start_date_neg'],
                indicator
            )
        elif self.model_name == 'PA':
            return PA(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], self.constants['last_year_start_date_neg'],
                indicator
            )
        elif self.model_name == 'GU':
            return GU(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], self.constants['last_year_start_date_neg'],
                indicator
            )
        elif self.model_name == 'DOR':
            return DOR(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['L_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], self.constants['last_year_start_date_neg'],
                indicator
            )
        elif self.model_name == 'DR':
            return DR(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], self.constants['last_year_start_date_neg'],
                indicator
            )
        elif self.model_name == 'FP':
            return FP(
                *parameters,
                self.variables['years'], self.constants['weather'], self.constants['flower'],
                self.constants['year_column'],
                self.constants['T_mean_column'], self.constants['flower_date_column'],
                self.constants['max_error'], self.constants['max_date'], self.constants['last_year_start_date_neg'],
                indicator
            )
        else:
            raise NotImplementedError(
                "The construction of model '" + self.model_name + "' is not implemented in method 'EcoModels.construct_model()'")

    def func_ys(self, parameters):
        ys = self.construct_model(parameters, 'ys')
        return ys

    def func_RMSE(self, *tuple_parameters_set, **dict_parameters):
        if tuple_parameters_set:
            parameters_set, = tuple_parameters_set
        elif dict_parameters:
            parameters_set = [dict_parameters[key] for key in self.models_parameters[self.model_name]['keys']]
        else:
            raise ValueError("The input of 'self.func_RMSE' is empty")

        if self.algorithm_name == 'dual_annealing':
            RMSE = self.construct_model(parameters_set, 'RMSE')

            try:
                self.records['n_f_evals'] += 1
                self.records['xs'].append(list(parameters_set))
                self.records['fs'].append(RMSE)
            except:
                pass

            return RMSE
        elif self.algorithm_name == 'differential_evolution':
            #  The DE algorithm set the relative tolerance for convergence to 1e-6, and when this condition was met,
            #  the iteration of DE algorithm stops
            RMSE = self.construct_model(parameters_set, 'RMSE')
            return RMSE
        elif self.algorithm_name == 'PSO':
            if np.array(parameters_set).ndim == 1:
                RMSE = self.construct_model(parameters_set, 'RMSE')
                return RMSE
            else:
                error_set = []
                for parameters in parameters_set:
                    RMSE = self.construct_model(parameters, 'RMSE')
                    error_set.append(RMSE)

                self._train_tuning_history['parameters_history'].append(parameters_set.tolist())
                self._train_tuning_history['rmses_history'].append(error_set)

                return np.array(error_set)
        elif self.algorithm_name == 'CMAES':
            RMSE = self.construct_model(parameters_set, 'RMSE')
            return RMSE,
        elif self.algorithm_name == 'bayesian_opt':
            RMSE = self.construct_model(parameters_set, 'RMSE')
            return -RMSE
        else:
            raise NotImplementedError(
                "The designated return for algorithm '" + self.algorithm_name + "' is not implemented in method 'EcoModels.func_RMSE()'")

    def func_AIC_NSE(self, parameters):
        AIC, NSE = self.construct_model(parameters, 'AIC&NSE')
        return AIC, NSE

    def load_models_parameters(self, models_parameters_file_path):
        with open(models_parameters_file_path, 'r') as file_to_read:
            lines = file_to_read.readlines()
            for line in lines:
                line_list = re.split('[:=]', line)
                for i, element in enumerate(line_list):
                    line_list[i] = element.replace(' ', '')
                line_list[-1] = line_list[-1].replace('\n', '')
                if line_list[0] == 'params':
                    try:
                        self.models_parameters[model_name]['keys'] = line_list[-1].split(',')
                    except NameError:
                        print(
                            " Error: The first row of each model's parameters data in models' parameters file (" + models_parameters_file_path + ") must be the model's name or abbreviation!")
                        break
                elif line_list[0] == 'min':
                    try:
                        self.models_parameters[model_name]['min'] = list(eval(line_list[-1]))
                    except NameError:
                        print(
                            " Error: The first row of each model's parameters data in models' parameters file (" + models_parameters_file_path + ") must be the model's name or abbreviation!")
                        break
                elif line_list[0] == 'max':
                    try:
                        self.models_parameters[model_name]['max'] = list(eval(line_list[-1]))
                    except NameError:
                        print(
                            " Error: The first row of each model's parameters data in models' parameters file (" + models_parameters_file_path + ") must be the model's name or abbreviation!")
                        break
                elif line_list[0] == 'init':
                    try:
                        self.models_parameters[model_name]['init'] = list(eval(line_list[-1]))
                    except NameError:
                        print(
                            " Error: The first row of each model's parameters data in models' parameters file (" + models_parameters_file_path + ") must be the model's name or abbreviation!")
                        break
                elif line_list[0] == '':
                    pass
                else:
                    model_name = line_list[0]
                    self.models_parameters[model_name] = {}

    def load_data(self, weather_data_file_path, flower_data_file_path):
        self.constants['weather'] = pd.read_csv(weather_data_file_path, index_col=0)
        self.constants['weather'] = self.constants['weather'].astype({'year': 'int32',
                                                                      'month': 'int32',
                                                                      'day': 'int32'
                                                                      })

        flowers_df = pd.read_csv(flower_data_file_path)
        if 'phenophase' in flowers_df.columns:
            phenophases_name = list(set(flowers_df['phenophase']))
            if self.phenophase_name in phenophases_name:
                flowers_df = flowers_df[flowers_df['phenophase'] == self.phenophase_name]
                flowers_df.drop(labels='phenophase', axis=1, inplace=True)
            else:
                raise ValueError(
                    "There is no match for [phenophase_name], the input of class 'EcoModels', in flower data file ("
                    + flower_data_file_path + "): " + self.phenophase_name + ". [phenophase_name] should be in " + str(
                        phenophases_name))

            flowers_name = list(set(flowers_df['species']))
            if self.species_name in flowers_name:
                self.constants['flower'] = flowers_df[flowers_df['species'] == self.species_name].copy().reset_index(
                    drop=True, inplace=False)
            else:
                raise ValueError(
                    "There is no match for [species_name], the input of class 'EcoModels', in flower data file ("
                    + flower_data_file_path + "): " + self.species_name + ". [species_name] should be in " + str(
                        flowers_name))
        else:
            self.constants['flower'] = flowers_df.drop(labels='Unnamed: 0', axis=1, inplace=False)

        self.constants['weather_years'] = set(self.constants['weather'][self.constants['year_column']])
        self.constants['flower_years'] = set(self.constants['flower'][self.constants['year_column']])
        self.constants['weather_flower_years'] = sorted(
            list(self.constants['weather_years'].intersection(self.constants['flower_years'])))
        self.constants['weather_years'] = sorted(list(self.constants['weather_years']))
        self.constants['flower_years'] = sorted(list(self.constants['flower_years']))
        self.constants['flower_name'] = list(set(self.constants['flower']['species']))

    def train(self, **kwargs):
        self._train_tuning_history = {
            'parameters_history': [],
            'rmses_history': [],
            'best_parameters_history': [],
            'best_RMSE_history': []
        }

        # Optimize models' parameters using different algorithms
        if self.algorithm_name == 'dual_annealing':
            self.records = {
                'n_f_evals': 0,
                'xs': [],
                'fs': []
            }

            parameters_bounds = list(
                zip(self.models_parameters[self.model_name]['min'], self.models_parameters[self.model_name]['max']))

            """
            The objective function's input generated by 'dual_annealing' is a 1-D np.ndarray.
            If `no_local_search` is set to True, a traditional Generalized Simulated Annealing will be
            performed with no local search strategy applied.
            """
            self._train_model = dual_annealing(
                self.func_RMSE, bounds=parameters_bounds, seed=self.algorithm_parameters['algorithm_RandomSeed'],
                x0=self.models_parameters[self.model_name]['init'], maxiter=self.algorithm_parameters['max_iter'],
                no_local_search=True, args=tuple(kwargs.values()))

            self._train_RMSE = self._train_model.fun
            self._optimal_parameters = list(self._train_model.x)

            # There are a few evaluations of objective function in `energy_state.reset` of `dual_annealing` before the process of optimization.
            population_size = 2 * len(parameters_bounds)
            n_f_evals_reset = self.records['n_f_evals'] - population_size * self.algorithm_parameters['max_iter']
            if n_f_evals_reset < 1:
                raise ValueError(
                    "The estimated number of objective function evaluations is larger than the actual one!")

            for i in range(0, self.algorithm_parameters['max_iter']):
                self._train_tuning_history['parameters_history'].append(self.records['xs'][
                                                                        n_f_evals_reset + i * population_size: n_f_evals_reset + (
                                                                                    i + 1) * population_size])
                self._train_tuning_history['rmses_history'].append(self.records['fs'][
                                                                   n_f_evals_reset + i * population_size: n_f_evals_reset + (
                                                                               i + 1) * population_size])
                try:
                    if min(self._train_tuning_history['rmses_history'][i]) < best_RMSE:
                        best_RMSE = min(self._train_tuning_history['rmses_history'][i])
                        best_index = self._train_tuning_history['rmses_history'][i].index(best_RMSE)
                        best_parameters = self._train_tuning_history['parameters_history'][i][best_index]
                except:
                    best_RMSE = min(self._train_tuning_history['rmses_history'][i])
                    best_index = self._train_tuning_history['rmses_history'][i].index(best_RMSE)
                    best_parameters = self._train_tuning_history['parameters_history'][i][best_index]
                self._train_tuning_history['best_parameters_history'].append(best_parameters)
                self._train_tuning_history['best_RMSE_history'].append(best_RMSE)

            del self.records
        elif self.algorithm_name == 'differential_evolution':
            # Define `callback_DE` function to record the history of optimized variables and that of objective function's value
            def callback_DE(xk, convergence):
                # 'xk' and 'f' are the variables and the value of objective function in each iteration.
                # If the parameter 'polish' in 'differential_evolution' is set to 'True', the best variables and
                # objective function's value returned by 'differential_evolution' will be diifferent with 'xk' and 'f'
                # in the last iteration.
                f = self.func_RMSE(xk)
                self._train_tuning_history['parameters_history'].append(None)
                self._train_tuning_history['rmses_history'].append(None)
                self._train_tuning_history['best_parameters_history'].append(list(xk))
                self._train_tuning_history['best_RMSE_history'].append(f)

            parameters_bounds = list(
                zip(self.models_parameters[self.model_name]['min'], self.models_parameters[self.model_name]['max']))

            # The objective function's input generated by 'differential_evolution' is a 1-D np.ndarray.
            self._train_model = differential_evolution(
                self.func_RMSE, bounds=parameters_bounds, maxiter=self.algorithm_parameters['max_iter'],
                seed=self.algorithm_parameters['algorithm_RandomSeed'], disp=True, polish=False, tol=1e-6,
                callback=callback_DE, args=tuple(kwargs.values()))

            self._train_RMSE = self._train_model.fun
            self._optimal_parameters = list(self._train_model.x)
        elif self.algorithm_name == 'PSO':
            parameters_dimensions = len(self.models_parameters[self.model_name]['init'])
            parameters_bounds = (
            self.models_parameters[self.model_name]['min'], self.models_parameters[self.model_name]['max'])

            num_particles = 20 if parameters_dimensions <= 5 else 200 if parameters_dimensions > 50 else 4 * parameters_dimensions

            # The objective function's input generated by 'PSO' is a n-D np.ndarray (n equals to num_particles).
            self._train_model = ps.single.GeneralOptimizerPSO(n_particles=num_particles,
                                                              dimensions=parameters_dimensions,
                                                              options=self.algorithm_parameters['PSO_Options'],
                                                              topology=self.algorithm_parameters['PSO_Topology'],
                                                              bounds=parameters_bounds)

            self._train_RMSE, self._optimal_parameters = self._train_model.optimize(
                self.func_RMSE, iters=self.algorithm_parameters['max_iter'], **kwargs)
            self._optimal_parameters = list(self._optimal_parameters)

            for i in range(0, self.algorithm_parameters['max_iter']):
                # :code: `self._train_tuning_history['parameters_history']` and :code: `self._train_tuning_history['rmses_history']` are recorded in class function `func_RMSE`
                try:
                    if min(self._train_tuning_history['rmses_history'][i]) < best_RMSE:
                        best_RMSE = min(self._train_tuning_history['rmses_history'][i])
                        best_index = self._train_tuning_history['rmses_history'][i].index(best_RMSE)
                        best_parameters = self._train_tuning_history['parameters_history'][i][best_index]
                except:
                    best_RMSE = min(self._train_tuning_history['rmses_history'][i])
                    best_index = self._train_tuning_history['rmses_history'][i].index(best_RMSE)
                    best_parameters = self._train_tuning_history['parameters_history'][i][best_index]
                self._train_tuning_history['best_parameters_history'].append(best_parameters)
                self._train_tuning_history['best_RMSE_history'].append(best_RMSE)
        elif self.algorithm_name == 'CMAES':
            # The objective function's input generated by 'CMAES' is a list.
            HallofFame, LogBook, History = CMAES(self.func_RMSE, self.models_parameters[self.model_name]['init'],
                                                 self.models_parameters[self.model_name]['min'],
                                                 self.models_parameters[self.model_name]['max'],
                                                 self.algorithm_parameters['max_iter'], verbose=True, **kwargs)

            self._train_RMSE = HallofFame[0].fitness.values[0]
            self._optimal_parameters = list(HallofFame[0])

            self._train_tuning_history['parameters_history'] = History['populations']
            self._train_tuning_history['rmses_history'] = History['fitnesses']
            self._train_tuning_history['best_parameters_history'] = History['best_xs']
            self._train_tuning_history['best_RMSE_history'] = History['best_fs']

        elif self.algorithm_name == 'bayesian_opt':
            parameters_bounds = dict(zip(self.models_parameters[self.model_name]['keys'],
                                         zip(self.models_parameters[self.model_name]['min'],
                                             self.models_parameters[self.model_name]['max'])
                                         )
                                     )

            # The objective function's input generated by 'bayesian_opt' is a dict/mapping with variables' keys and values.
            max_iter = self.algorithm_parameters['max_iter']
            iter_ratio = self.algorithm_parameters['bayesian_opt_IterRatio']
            self._train_model = BayesianOptimization(
                f=self.func_RMSE,
                pbounds=parameters_bounds,
                verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                random_state=self.algorithm_parameters['algorithm_RandomSeed'],
            )
            self._train_model.maximize(init_points=int(iter_ratio * max_iter),
                                       n_iter=int((1 - iter_ratio) * max_iter))
            params = self._train_model.max['params']

            self._train_RMSE = -self._train_model.max['target']
            self._optimal_parameters = [params[key] for key in self.models_parameters[self.model_name]['keys']]

            for i, res in enumerate(self._train_model.res):
                self._train_tuning_history['parameters_history'].append(
                    [res['params'][key] for key in self.models_parameters[self.model_name]['keys']])
                self._train_tuning_history['rmses_history'].append(-res['target'])
                if i <= int(
                        self.algorithm_parameters['bayesian_opt_IterRatio'] * self.algorithm_parameters['max_iter']):
                    best_RMSE = self._train_tuning_history['rmses_history'][i]
                    best_parameters = self._train_tuning_history['parameters_history'][i]
                else:
                    if self._train_tuning_history['rmses_history'][i] < best_RMSE:
                        best_RMSE = self._train_tuning_history['rmses_history'][i]
                        best_parameters = self._train_tuning_history['parameters_history'][i]
                self._train_tuning_history['best_parameters_history'].append(best_parameters)
                self._train_tuning_history['best_RMSE_history'].append(best_RMSE)
        else:
            raise NotImplementedError(
                "The designated process of algorithm '" + self.algorithm_name + "' is not implemented in method 'EcoModels.train()'")

        # Use `sort` function to rank part of parameters in optimization result,
        # which was listed in 'self.constants['parameters_to_be_sorted']',
        # in order to match the processing in 'base_functions.py'
        parameters_keys = self.models_parameters[self.model_name]['keys']
        for parameters_to_be_sorted in self.constants['parameters_to_be_sorted']:
            if set(parameters_to_be_sorted).issubset(parameters_keys):
                index_list = [parameters_keys.index(parameter_key) for parameter_key in parameters_to_be_sorted]
                values_sorted = sorted([self._optimal_parameters[index] for index in index_list])
                for i, index in enumerate(index_list):
                    self._optimal_parameters[index] = values_sorted[i]
            else:
                pass

        # Calculate the other metrics except RMSE, such as AIC and NSE
        self._train_AIC, self._train_NSE = self.func_AIC_NSE(self._optimal_parameters)
        
        

    def test(self, **kwargs):
        if self.algorithm_name == 'bayesian_opt':
            RMSE = self.func_RMSE(self._optimal_parameters)
            self._test_RMSE = -RMSE[0] if isinstance(RMSE, tuple) else -RMSE
        else:
            RMSE = self.func_RMSE(self._optimal_parameters)
            self._test_RMSE = RMSE[0] if isinstance(RMSE, tuple) else RMSE
        self._test_AIC, self._test_NSE = self.func_AIC_NSE(self._optimal_parameters)
        
    def train_and_test(self, **kwargs):

        weather_flower_years = np.array(self.constants['weather_flower_years'])
        flower_df = self.constants['flower'].copy()

        column_predictions_name = 'y_pred'
        column_labels_name = 'label'
        flower_df[column_labels_name] = ''

        train_years, test_years = train_test_split(weather_flower_years,
                                                                 train_size=0.7,
                                                                 random_state=self.algorithm_parameters['split_seed'])

        self.variables['years'] = train_years
        self.train()
        self._train_best_parameters = self._optimal_parameters
        self._train_tuning_history = pd.DataFrame(self._train_tuning_history)

        self.variables['years'] = test_years
        self.test()
        self.variables['years'] = weather_flower_years
        prediction_df = pd.DataFrame({'year': weather_flower_years, 'prediction': self.func_ys(self._optimal_parameters)})
        flower_df = pd.merge(flower_df, prediction_df, how='outer', on='year')

        for year in train_years:
            flower_df.loc[flower_df[self.constants['year_column']] == year, column_labels_name] = 'train'
        for year in test_years:
            flower_df.loc[flower_df[self.constants['year_column']] == year, column_labels_name] = 'test'
        self._predictions_df = flower_df.copy()


    def fit(self, **kwargs):
        self.variables['years'] = self.constants['weather_flower_years']
        self.train()
        self._fit_best_parameters = self._optimal_parameters
        self._fit_RMSE = self._train_RMSE
        self._fit_AIC = self._train_AIC
        self._fit_NSE = self._train_NSE
        self._fit_tuning_history = pd.DataFrame(self._train_tuning_history)

    def predict(self, **kwargs):
        self.variables['years'] = self.constants['weather_years'][1:]
        self._predictions = self.func_ys(self._fit_best_parameters)

    def predictions_to_DataFrame(self):
        predictions_df = pd.DataFrame({self.constants['year_column']: self.constants['weather_years'][1:],
                                       'y_pred': self._predictions})
        predictions_df = predictions_df.merge(self.constants['flower'], how='left', on=[self.constants['year_column']],
                                              suffixes=['_pred', '_true'])
        predictions_df['date_err'] = predictions_df[self.constants['flower_date_column']] - predictions_df['y_pred']
        predictions_df['species'].fillna(self.species_name, inplace=True)
        self._predictions_df = predictions_df

    @property
    def train_best_parameters(self):
        return self._train_best_parameters

    @property
    def train_RMSE(self):
        return self._train_RMSE

    @property
    def train_AIC(self):
        return self._train_AIC

    @property
    def train_NSE(self):
        return self._train_NSE

    @property
    def test_RMSE(self):
        return self._test_RMSE

    @property
    def test_AIC(self):
        return self._test_AIC

    @property
    def test_NSE(self):
        return self._test_NSE

    @property
    def predictions_df(self):
        return self._predictions_df

    @property
    def fit_best_parameters(self):
        return self._fit_best_parameters

    @property
    def fit_RMSE(self):
        return self._fit_RMSE

    @property
    def fit_AIC(self):
        return self._fit_AIC

    @property
    def fit_NSE(self):
        return self._fit_NSE

    @property
    def fit_tuning_history(self):
        return self._fit_tuning_history

    @property
    def train_tuning_history(self):
        return self._train_tuning_history

    @property
    def predictions_df(self):
        return self._predictions_df

    @property
    def predict_history(self):
        predict_results = pd.DataFrame({'predictions': self._predictions}, index=self.constants['weather_years'][1:]).T
        predict_results.insert(0, 'phenophase', self.phenophase_name)
        predict_results.insert(0, 'species', self.constants['flower_name'])
        predict_results.insert(0, 'algorithm', self.algorithm_name)
        predict_results.insert(0, 'model', self.model_name)
        self._predict_history = predict_results
        return self._predict_history
