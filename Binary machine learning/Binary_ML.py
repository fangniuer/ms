# -*- coding: utf-8 -*-
# @Time    : 2020/08/15 10:10
# @Author  : Wujun Dai
# @Email   : wjdainefu@126.com
# @File    : Binary_ML.py
# @Software: PyCharm

import datetime
import numpy as np
import os
import pandas as pd

from pycaret.classification import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_data(
        weather_data_file_path, phenological_data_file_path,
        weather_features, phenophases_set, species_set,
        year_col='year', month_col='month', day_col='day', date_col='date'
):
    weather_df = pd.read_csv(weather_data_file_path, index_col=0, dtype={year_col: 'int32',
                                                                         month_col: 'int32',
                                                                         day_col: 'int32',
                                                                         date_col: 'int32'})
    phe_df = pd.read_csv(phenological_data_file_path, dtype={year_col: 'int32',
                                                             month_col: 'int32',
                                                             day_col: 'int32',
                                                             date_col: 'int32'})

    # Check the input of `phenophases_set`, `species_set` and `weather_features`
    weather_df_columns = set(weather_df.columns)
    phenophase_names = set(phe_df['phenophase'])
    species_names = set(phe_df['species'])

    if set(weather_features).issubset(weather_df_columns):
        pass
    else:
        feature_error = list(set(weather_features) - weather_df_columns)
        raise ValueError(
            "There is no match for the input of [weather_features] in the columns of weather data file ({}): {}. The input of [weather_features] should be a subset of {}.".format(
                weather_data_file_path, feature_error, list(weather_df_columns)))
    if set(phenophases_set).issubset(phenophase_names):
        pass
    else:
        phenophase_error = list(set(phenophases_set) - phenophase_names)
        raise ValueError(
            "There is no match for the input of [phenophases_set] in the phenophase column of phenological data file ({}): {}. The input of [phenophases_set] should be a subset of {}.".format(
                phenological_data_file_path, phenophase_error, list(phenophase_names)))
    if set(species_set).issubset(species_names):
        pass
    else:
        species_error = list(set(species_set) - species_names)
        raise ValueError(
            "There is no match for the input of [species_set] in the species column of phenological data file ({}): {}. The input of [species_set] should be a subset of {}.".format(
                phenological_data_file_path, species_error, list(species_names)))

    # Check the years in weather data whether are consecutive or not
    weather_years = set(weather_df[year_col])
    weather_years_missing = set(range(min(weather_years), max(weather_years) + 1)) - weather_years
    if weather_years_missing:
        raise ValueError("The weather data in years {} are missing!".format(list(weather_years_missing)))
    else:
        pass
    return weather_df, phe_df

def select_data(df, years, year_col='year',date_col='date', bloom_col='bloom'):
    '''

    Parameters
    ----------
    dataset_df : pandas.DataFrame
        The generated dataset with year, date, weather features and bloom columns.
    years : list
        The years for dataset to be selected.
    year_col : str, optional
        The name of year column in weather data and flower data. The default is 'year'.
    date_col : str, optional
        The name of date column in weather data and flower data. The default is 'date'.
    bloom_col : str, optional
        The name of bloom column in the generated dataset. The default is 'bloom'.

    Returns
    -------
    data_selected_date : pandas.DataFrame
        The columns of year and date for selected dataset.
    data_selected : pandas.DataFrame
        The columns of weather features and bloom for selected dataset.

    '''

    data_selected = df[df[year_col].isin(years)].reset_index(drop=True).dropna(axis=0, how='any')
    data_selected_date = data_selected.loc[:, [year_col, date_col]].copy()
    data_selected = data_selected.drop(columns=[year_col, date_col])
    data_selected[bloom_col] = data_selected[bloom_col].astype(int)
    return data_selected_date, data_selected

def generate_dataset(
        weather_df, weather_features, species_obs_df, random_state,
        window_days=30, pos_sample_size=30, interpolate=False, dataset_dir='Dataset/', unseen_ratio=0.3,
        year_col='year', date_col='date', bloom_col='bloom'):

    """

    Parameters
    ----------
    weather_df : pandas.DataFrame
        The weather data in a certain place.
    weather_features : list
        The selected columns' names in weather data, and the corresponding columns will be used as features in machine learning.
    species_obs_df : pandas.DataFrame
        The flower data for a certain species in a specific phenophase.
    random_state ： int
        All observed years were divided into training set and test set according the random_state
    obs_years : list
        The intersection of years in weather data and flower data, according to which the values in bloom column  will be generated.
    dataset_file_path : str
        The path of file used to save dataset.
    window_days : int, default = 30
        The length of time window.
        The interval days between two consecutive data segments from which dataset is generated. It is determined by the maximum difference of date for the same species and phenophase. The default is 40.
    pos_sample_size : int, default = 30
        Number of positive samples created.
    interpolate : Boolean, default = False
        If it is True, the null values in weather data will be filled by using `interpolate` method for DataFrame. The default is False.
    dataset_dir ;  str
        File directory for created dataset
    unseen_ratio; float
        All observed years were divided into training set and test set according the unseen_ratio
    year_col : str, optional
        The name of year column in weather data and flower data. The default is 'year'.
    month_col : str, optional
        The name of month column in weather data and flower data. The default is 'month'.
    day_col : str, optional
        The name of day column in weather data and flower data. The default is 'day'.
    date_col : str, optional
        The name of date column in weather data and flower data. The default is 'date'.
    bloom_col : str, optional
        The name of bloom column in the generated dataset. The default is 'bloom'.

    Returns
    -------
    years_unseen : years for testing
    years_seen : years for training
    dataset : pandas.DataFrame
        The generated dataset with year, date, weather features and bloom columns.
    data_prediction : Data needed to predict the year of the meteorological data
    data_known : Data for all observed years. finalize_models()
    data_seen : Data for compare_models()
    data_unseen : Data for RMSE calculation

    """

    dataset_file_path = dataset_dir + f"{'-'.join(weather_features)}_window_days_{window_days}.csv"
    obs_years = sorted(list(set(species_obs_df[year_col])))
    obs_doy_mean = round((species_obs_df[date_col]).mean())
    weather_years = sorted(list(set(weather_df[year_col])))

    if os.path.exists(dataset_file_path):
        dataset_df = pd.read_csv(dataset_file_path)
    else:
        if set(obs_years).issubset(set(weather_years)):
            pass
        else:
            excess_years = list(set(weather_years) - set(obs_years))
            raise ValueError(f'{excess_years} have no meteorological data for {species_name}, {phenophase}')

        if os.path.isdir(dataset_dir):
            pass
        else:
            os.makedirs(dataset_dir)

        if interpolate:
            weather_df = weather_df.interpolate()  # The default method is 'linear'

        # Generate daily dataset according to window_days and weather_features
        dataset_df_columns = [year_col, date_col]
        weather_features_columns = []

        for weather_feature in weather_features:
            weather_feature_columns = [f'{weather_feature}_{day}' for day in range(1, window_days + 1)]
            weather_features_columns.extend(weather_feature_columns)
        dataset_df_columns.extend(weather_features_columns)
        dataset_df_columns.append(bloom_col)

        weather_df_index = list(weather_df.index)
        dataset_df = weather_df.loc[:weather_df_index[-window_days], [year_col, date_col]].copy()
        dataset_df = dataset_df.reindex(columns=dataset_df_columns)

        for i in weather_df_index[: 1 - window_days]:
            weather_feature_data = []
            for weather_feature in weather_features:
                weather_feature_data.extend(list(weather_df.loc[i: i + window_days - 1, weather_feature]))

            dataset_df.loc[i, weather_features_columns] = weather_feature_data

        # Save dataset_df as a CSV file
        dataset_df[[year_col, date_col]] = weather_df.loc[window_days - 1:,[year_col, date_col]].reset_index(drop=True)
        dataset_df.to_csv(dataset_file_path, index=False)

    data_prediction = dataset_df.copy()


    # Sample from specific columns in weather data using time window method.
    # The date of middle point of sampling segment is that the blooming date of this species minus window_days and plus 1.
    neg_sample_size =  pos_sample_size
    dataset_df_index = list(dataset_df.index)

    for year in obs_years:
        doy = species_obs_df.loc[species_obs_df[year_col] == year][date_col].values[0]
        date_bloom_index = list(dataset_df[(dataset_df[year_col] == year) & (dataset_df[date_col] == doy)].index)[0]
        dataset_df.loc[date_bloom_index - neg_sample_size: date_bloom_index - 1, [year_col, bloom_col]] = [[year, 0]] * neg_sample_size

        if dataset_df_index[-1] - date_bloom_index + 1 < pos_sample_size:
            dataset_df.loc[date_bloom_index:, [year_col, bloom_col]] = [[year, 1]] * (dataset_df_index[-1] - date_bloom_index + 1)
        else:
            dataset_df.loc[date_bloom_index: date_bloom_index + pos_sample_size - 1, [year_col, bloom_col]] = [[year, 1]] * pos_sample_size

    # Split obs_years to seen part and unseen part
    years_unseen, years_seen = train_test_split(obs_years, train_size=unseen_ratio, random_state=random_state)


    data_known_date, data_known = select_data(dataset_df, obs_years)
    data_seen_date, data_seen = select_data(dataset_df, years_seen)
    data_unseen_date, data_unseen = select_data(dataset_df, years_unseen)
    # data_predict_date, data_predict = select_data(dataset_df, weather_years[1:])

    # Prepare data for Pycaret
    # dataset_df contains nan values, converting float to integer
    data_prediction = dataset_df.copy()
    data_prediction_date = data_prediction.loc[:, [year_col, date_col]].copy()
    data_prediction = data_prediction.drop(columns=[year_col, date_col])


    return years_unseen, years_seen, data_prediction_date, data_prediction,\
           data_known_date, data_known, data_seen_date, data_seen, data_unseen_date, data_unseen


def get_model_name(model, model_dict):
    model_str = str(model).spilt("(")[0]

    # For catboost model, create_model() will return: <catboost.core.CatBoostClassifier at 0x25a4c5cba60>
    # knn model will return: KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski')
    if 'catboost' in model_str:
        model_str = 'CatBoostClassifier'

    model_name = model_dict.get(model_str)
    return model_name

def binary_to_date(binary_results_df, years, phenophase_fluctuation, judge_period_days,
                   obs_doys, zero_ratio, year_col='year', date_col='date'):
    '''

    Parameters
    ----------
    binary_results_df : pandas.DataFrame
        The results of binary classification predicted by finalized model.
    years : list
        The selected years in dataset.
    window_days : int
        The length of time window.
    phenophase_fluctuation : int
        The fluctuation of phenophase date for a certain species.
    judge_period_days : int
        The length of time window in binary results used to judge whether flower bloom or not.
    zero_ratio : float
        The ratio of zero in the values whose size is `judge_period_days`.
    date_col : str, optional
        The name of date column in weather data and flower data. The default is 'date'.

    Returns
    -------
    pandas.DataFrame
        The predictions of date in each year.

    '''
    date_predictions = []
    # initial_value = judge_period_days
    initial_value = zero_ratio
    for year in years:
        Labels = list(binary_results_df[binary_results_df[year_col] == year]['Label'].values)
        dates = list(binary_results_df[binary_results_df[year_col] == year][date_col].values)
        flag = 0
        # judge_period_days = initial_value
        zero_ratio = initial_value
        # criterion = judge_period_days / 2
        criterion = 0.5
        # while flag == 0 and judge_period_days >= criterion:
        while flag == 0 and zero_ratio <= criterion:
            for i in range(0, len(Labels) - judge_period_days + 1):
                # if Labels[i : i + judge_period_days] == [1]*judge_period_days:
                if (Labels[i] == 1) and ((judge_period_days - sum(Labels[i: i + judge_period_days])) <= (
                        judge_period_days * zero_ratio)):
                    date_prediction = dates[i] - 1
                    if (date_prediction >= min(obs_doys) - phenophase_fluctuation) and (
                            date_prediction <= max(obs_doys) + phenophase_fluctuation):
                        date_predictions.append(date_prediction)
                        flag = 1
                        break
            # judge_period_days -= 1
            zero_ratio += 0.1
        if flag == 0:
            date_predictions.append(max(obs_doys) + phenophase_fluctuation)
    doy_result = pd.DataFrame({year_col: years, 'date_predictions': date_predictions})
    return doy_result



# Predict using the final models, convert binary results to predictions of date, calculate the corresponding RMSE, and save the predictions as well as RMSEs
def get_metric_pre_rmse(species_obs_df, obs_doys, species_name,
                            species_file_path, filename_suffix, n,
                            model, data, data_name, data_date, years,
                            phenophase_fluctuation, judge_period_days, zero_ratio,
                            year_col='year', date_col='date', save_bool=True, RMSE_bool=True, index_bool=False):
    if RMSE_bool:
        display_grid = get_config('display_container')
        train_acc = display_grid[n].iloc[k_fold, 0]
        predict_model(model, verbose=verbose_bool)
        test_acc = pull().iloc[0, 1] # pull() or get_config('display_container')[-1]

    final_model = finalize_model(model)
    data_label_predictions = predict_model(final_model, data=data)
    data_binary_results = pd.concat([data_date, data_label_predictions], axis=1, join='outer')

    data_doy_predictions = binary_to_date(data_binary_results, years, phenophase_fluctuation, judge_period_days,
                                           obs_doys, zero_ratio, year_col, date_col)
    data_doy_results = pd.merge(species_obs_df, data_doy_predictions, how='outer', on=year_col, sort=True)

    if RMSE_bool:
        data_doy_results['error'] = data_doy_results[date_col] - data_doy_results['date_predictions']
        if save_bool:
            data_doy_results.to_csv(f"{species_file_path}{data_name}_doy-{filename_suffix}",
                                    index=index_bool, encoding='utf_8_sig')
        errors = list(data_doy_results[data_doy_results['error'].notnull()]['error'].values)
        RMSE = round(np.sqrt(sum(np.power(errors, 2)) / len(errors)), 3)
        return train_acc, test_acc, RMSE
    else:
        data_doy_results['species'] = species_name
        data_binary_results.to_csv(f"{species_file_path}{data_name}_binary-{filename_suffix}", index=index_bool)
        data_doy_results.to_csv(f"{species_file_path}{data_name}_doy-{filename_suffix}",
                                    index=index_bool, encoding='utf_8_sig')


if __name__ == "__main__":
    """
    Parameters that can be customized
    """
    weather_data_file_path = './data/weather_harbin.csv'
    phenological_data_file_path = './data/phe_cleaned.csv'
    result_dir = './tmp'
    rmse_file_name = 'rmse_ensemble.csv'
    compare_grid_file_name = 'compare_grid.csv'

    weather_features = ['mean_TEM']
    # ['PHO', 'daylong_PRE', 'mean_PRS', 'mean_RHU', 'min_RHU', 'SSD', 'mean_TEM', 'max_TEM', 'min_TEM', 'mean_WIN']
    phenophases_set = ['FFD']  # ['LUD', 'FFD', 'LFD', 'EOS']
    species_set = ['紫丁香']  # ['紫丁香', '樟子松', '东北山梅花', '红花锦鸡儿', '山杏', '金银忍冬']

    random_state = 303
    unseen_ratio = 0.3  # The sample proportion of all observed years divided into test years
    train_ratio = 0.8  # The ratio of data_seen as training set in Pycaret.
    # All data_seen are used as a training set when finalize_model() is executed
    phenophase_fluctuation = 30 # If the maximum phenological date later than  December 1, this value should be adjusted

    window_days_range = range(10, 30+1, 10)  # Format: 1. range(window_days_lowerbound, window_days_upperbound + 1, step_size)
    #         2. List type data of window days, such as [7, 8]

    num_top_models = 18
    k_fold = 5
    tune_iteration = 100


    pos_sample_size_set = range(50,50 + 1, 10)  # Should be less than (365 - 30)/2
    # pos_sample_size = 60
    # Calculate RMSE on a test set
    create_on_seen = True
    tune_hyp = False
    emsemble_within = False
    blend_within = False
    stack_within = False
    ensemble_across_model = False
    ensemble_across_method = 'Blending'  # [ 'Bagging', 'Boosting', 'Blending', 'Stacking']

    weather_year_prediction, tune_hyp_prediction = False,  False  #  True or False
    # model_input = None  # It should be 'None' or a list object contains one of the following:
    model_input = ['et']
    # model_input = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf',
    #                  'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost']

    # # The model_input is set to None when performing a comparison of the 18 classification models
    # model_input = None

    '''Executive prediction'''
    # create_on_seen = False
    # tune_hyp = False
    # create_on_known = True  # True
    # tune_on_known = True  # True
    # model_input = ['et']  # It should be 'None' or a subset of the following:
    # ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf',
    #                  'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost']
    # When `model_input` is set to 'None', the `compare_models` process will be implemented.

    # checking boosting conflict with estimators

    '''Ensemble'''
    # Increasing the number of estimators can sometimes improve results.

    select_best_for_prediction = False
    best_model_metric = 'Accuracy'  # 'RMSE' or 'Accuracy'
    tune_pre = False
    n_estimators = 3  # Number of estimators for ensemble_model()
    plot_feature = False
    interpret_summary = False
    save_top_models_grid = False  # Save output of Compare Baseline
    save_bin_doy_bool = False  # Save binary and doy results

    judge_period_days = 7
    zero_ratio = 0.2  # The minimum value is 0, and the maximum value is 0.5
    interpolate = False
    verbose_bool = False
    year_col = 'year'
    month_col = 'month'
    day_col = 'day'
    date_col = 'date'
    bloom_col = 'bloom'
    optimize_metric = 'Accuracy'  # ['Accuracy', 'AUC', 'Recall', 'Precision', 'F1']
    stack_method = 'auto'
    stack_meta_method = 'lr'  # ['catboost', 'xgboost', 'lr', etc.]
    blend_method = 'soft'  # ['soft', 'hard']
    plot_feature_name = 'Feature Importance.png'
    save_score_grid_bool = False
    save_rmse = True

    model_str = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf',
                 'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost']

    model_dict = {'ExtraTreesClassifier': 'et',
                  'GradientBoostingClassifier': 'gbc',
                  'RandomForestClassifier': 'rf',
                  'LGBMClassifier': 'lightgbm',
                  'XGBClassifier': 'xgboost',
                  'AdaBoostClassifier': 'ada',
                  'DecisionTreeClassifier': 'dt',
                  'RidgeClassifier': 'ridge',
                  'LogisticRegression': 'lr',
                  'KNeighborsClassifier': 'knn',
                  'GaussianNB': 'nb',
                  'SGDClassifier': 'svm',
                  'SVC': 'rbfsvm',
                  'GaussianProcessClassifier': 'gpc',
                  'MLPClassifier': 'mlp',
                  'QuadraticDiscriminantAnalysis': 'qda',
                  'LinearDiscriminantAnalysis': 'lda',
                  'CatBoostClassifier': 'catboost',
                  'BaggingClassifier': 'Bagging'}

    models_to_interpret = ["<class 'catboost.core.CatBoostClassifier'>",
                           "<class 'sklearn.ensemble._gb.GradientBoostingClassifier'>",
                           "<class 'sklearn.ensemble._forest.ExtraTreesClassifier'>",
                           "<class 'lightgbm.sklearn.LGBMClassifier'>",
                           "<class 'sklearn.ensemble._forest.RandomForestClassifier'>",
                           "<class 'sklearn.tree._classes.DecisionTreeClassifier'>"]
    models_to_plot = ["<class 'sklearn.ensemble._gb.GradientBoostingClassifier'>",
                      "<class 'sklearn.ensemble._forest.ExtraTreesClassifier'>",
                      "<class 'lightgbm.sklearn.LGBMClassifier'>",
                      "<class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>",
                      "<class 'xgboost.sklearn.XGBClassifier'>",
                      "<class 'sklearn.linear_model._logistic.LogisticRegression'>",
                      "<class 'sklearn.ensemble._forest.RandomForestClassifier'>",
                      "<class 'sklearn.linear_model._ridge.RidgeClassifier'>",
                      "<class 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'>",
                      "<class 'sklearn.linear_model._stochastic_gradient.SGDClassifier'>",
                      "<class 'sklearn.tree._classes.DecisionTreeClassifier'>"]
    """
    Automatically generated parameters
    """
    if model_input:
        if set(model_input).issubset(set(model_dict.values())):
            compare_baseline = False
        else:
            raise ValueError(f"Unsupported model:{set(model_dict.values()) - set(model_input)}")
    else:
        compare_baseline = True
        create_on_seen = False
        save_rmse = False


    """
    Load weather and phenological data
    """
    weather_df, phe_df = load_data(weather_data_file_path, phenological_data_file_path,
                                      weather_features, phenophases_set, species_set,
                                      year_col, month_col, day_col, date_col)
    weather_years = sorted(list(set(weather_df[year_col])))

    """
    Iterations for sample_size,phenophase and species
    """
    unseen_rmse_results = pd.DataFrame()
    compare_grid_results = pd.DataFrame()
    for pos_sample_size in pos_sample_size_set:
        for phenophase in tqdm(phenophases_set):
            phenophase_file_path = result_dir + '/' + phenophase + '/'
            for species_name in species_set:
                species_file_path = phenophase_file_path + species_name + '/'

                # Check the file path whether exists or not. If not, create it
                if os.path.isdir(species_file_path):
                    pass
                else:
                    os.makedirs(species_file_path)

                start_time = datetime.datetime.now()
                species_obs_df = phe_df.query('phenophase == @phenophase and species == @species_name').copy().reset_index(drop=True)
                obs_doys = species_obs_df[date_col].values

                for window_days in tqdm(window_days_range):
                    filename_suffix = f"{phenophase}_{species_name}-window_days_{window_days}-{'|'.join(weather_features)}-sample_size_{pos_sample_size}.csv"

                    # Generate dataset for pycaret
                    years_unseen, years_seen, data_prediction_date, data_prediction, data_known_date,data_known, \
                    data_seen_date, data_seen, data_unseen_date, data_unseen = generate_dataset(weather_df, weather_features, species_obs_df, random_state,
                                                    window_days, pos_sample_size, interpolate, 'Dataset/', unseen_ratio,
                                                    year_col, date_col, bloom_col)

                    """
                    Training and compare on data_seen(split by unseen_ratio)
                    """
                    # Setting up environment in pycaret
                    phe_clf_seen = setup(data=data_seen, target=bloom_col, train_size=train_ratio,
                                         session_id=random_state, html=False, silent=True, verbose=verbose_bool)

                    # Comparing Baseline: compare all the available models in the model library in pycaret
                    if compare_baseline:
                        top_models = compare_models(fold=k_fold, n_select=num_top_models, turbo=False, verbose=verbose_bool)

                        display_grid = get_config('display_container')
                        compare_grid = pull()

                        # Save num_top_modeds score_grid
                        if save_top_models_grid:
                            for i, model in enumerate(top_models):
                                estimator_name = compare_grid.Model[i]
                                display_grid[i].to_csv(species_file_path + f"compare-model_{i + 1}-{estimator_name}_score_grid-{filename_suffix}")


                        for n, model in enumerate(top_models):
                            train_acc, test_acc, rmse = get_metric_pre_rmse(
                                species_obs_df, obs_doys, species_name, species_file_path + 'best',
                                filename_suffix, n, model, data_unseen, 'data_unseen', data_unseen_date,
                                years_unseen, phenophase_fluctuation, judge_period_days, zero_ratio, year_col,
                                date_col, save_bool=save_bin_doy_bool)

                            compare_grid.loc[n, 'RMSE'] = rmse
                            compare_grid.loc[n, 'train_acc'] = train_acc
                            compare_grid.loc[n, 'test_acc'] = test_acc
                        compare_grid = compare_grid.sort_values('RMSE', ascending=True)
                        set_config('display_container', [])
                        compare_grid = pd.concat([compare_grid, pd.DataFrame({'phenophase': phenophase,
                                                                              'species': species_name,
                                                                              'window_days': window_days,
                                                                              'pos_sample_size': pos_sample_size,
                                                                              'seed': random_state,
                                                                              'weather_features': '|'.join(weather_features)
                                                                              }, index=compare_grid.index)], axis=1)
                        compare_grid_results = compare_grid_results.append(compare_grid)


                        if select_best_for_prediction:
                            if best_model_metric == 'Accuracy':
                                choose = compare_grid[compare_grid[best_model_metric] == max(compare_grid[best_model_metric])]
                            elif best_model_metric == 'RMSE':
                                choose = compare_grid[compare_grid[best_model_metric] == min(compare_grid[best_model_metric])]

                            model_best = top_models[choose[choose['TT (Sec)']==min(choose['TT (Sec)']).Model.values[0]]]
                            model_best_str = get_model_name(model_best, model_dict)

                            phe_clf_prediction = setup(data=data_known, target=bloom_col, train_size=train_ratio,
                                         session_id=random_state, html=False, silent=True, verbose=verbose_bool)

                            pre_model = create_model(model_best_str, fold=k_fold, verbose=verbose_bool)

                            if tune_pre:
                                tuned_pre_model = tune_model(pre_model, fold=k_fold, n_iter=tune_iteration,
                                                     choose_better=True, optimize=optimize_metric, verbose=verbose_bool)
                                final_pre_model = tuned_pre_model
                            else:
                                final_pre_model = pre_model

                            get_metric_pre_rmse(species_obs_df, obs_doys, species_name, species_file_path,
                                filename_suffix, 0, final_pre_model, data_prediction, 'prediction',
                                 data_prediction_date, weather_years[1:], phenophase_fluctuation,
                                 judge_period_days, zero_ratio, year_col, date_col,
                                save_bool=save_bin_doy_bool, RMSE_bool=False)




                    """
                    Create models
                    """
                    if create_on_seen:

                        created_models = []
                        tuned_models = []
                        bagged_models = []
                        boosted_models = []

                        blend_within_models = []
                        stack_within_models = []

                        estimator_names = []
                        estimators_list = []
                        estimators_within = [] # Set of optional models with the same name model

                        for i, name in enumerate(model_input):
                            created_model = create_model(estimator=name, fold=k_fold, verbose=verbose_bool)
                            estimator_names.append('created_' + name)
                            estimators_list.append(created_model)
                            estimators_within .append(created_model)
                            created_models.append(created_model)

                            # Tune Hyperparameters
                            if tune_hyp:
                                tuned_model = tune_model(created_model, fold=k_fold, n_iter=tune_iteration,
                                                         choose_better=True, optimize=optimize_metric, verbose=verbose_bool)
                                target_model = tuned_model
                                estimator_names.append('tuned_' + name)
                                estimators_list.append(tuned_model)
                                estimators_within.append(tuned_model)
                                tuned_models.append(tuned_model)
                            else:
                                target_model = created_model

                            # ensemble within the same type of model
                            if emsemble_within:
                                # available_ensemble_method = ['Bagging', 'Boosting']
                                # if ensemble_method not in available_ensemble_method:
                                #     ensembled_rmse = np.nan
                                bagged_model = ensemble_model(target_model, fold=k_fold, method='Bagging',
                                                                 n_estimators=n_estimators, verbose=verbose_bool)
                                estimator_names.append('bagged_' + name)
                                estimators_list.append(bagged_model)
                                estimators_within.append(bagged_model)
                                bagged_models.append(bagged_model)

                                boosting_unsupported = ['lda', 'qda', 'ridge', 'mlp', 'gpc', 'svm', 'knn', 'catboost']

                                if  name in boosting_unsupported:
                                    print(f"model {name} doesn't support 'Boosting'")
                                else:
                                    boosted_model = ensemble_model(target_model, fold=k_fold, method='Boosting',
                                                                 n_estimators=n_estimators, verbose=verbose_bool)
                                    estimator_names.append('boosted_' + name)
                                    estimators_list.append(boosted_model)
                                    estimators_within.append(boosted_model)
                                    boosted_models.append(boosted_model)

                            '''
                            Blending and Stacking within the same type of model(name)
                            '''
                            if blend_within:
                                blend_within_model = blend_models(estimators_within, fold=k_fold, choose_better=True,
                                                             optimize=optimize_metric, verbose=verbose_bool)
                                estimator_names.append('blend_within_' + name)
                                estimators_list.append(blend_within_model)
                                blend_within_models.append(blend_within_model)

                            if stack_within: # Stack within the same type of model (name)
                                stack_within_model = stack_models(estimators_within, fold=k_fold, choose_better=True,
                                                             optimize=optimize_metric, verbose=verbose_bool)
                                estimator_names.append('stack_within_' + name)
                                estimators_list.append(stack_within_model)
                                stack_within_models.append(stack_within_model)

                        '''
                        Bagging, Boosting, Blending and Stacking across different type of models
                        '''
                        if ensemble_across_model:
                            across_etimators_set = [created_models, tuned_models, bagged_models, boosted_models,
                                                    blend_within_models, stack_within_models]
                            prefix = ['created_', 'tuned_', 'bagged_', 'boosted_', 'blend_within_', 'stack_within_']

                            across_etimators_set = list(filter(None, across_etimators_set))

                            for ensemble_across_method in ensemble_across_method:

                                if ensemble_across_method == 'Bagging':
                                    for across_etimators in enumerate(across_etimators_set):
                                        bagged_across_model = ensemble_model(across_etimators, fold=k_fold, choose_better=True,
                                                                             method='Bagging', optimize=optimize_metric,
                                                                             verbose=verbose_bool)
                                        estimator_names.append('bagged_across')
                                        estimators_list.append(bagged_across_model)
                                elif ensemble_across_method == 'Boosting':
                                    for across_etimators in enumerate(across_etimators_set):
                                        boosted_across_model = ensemble_model(across_etimators, fold=k_fold, choose_better=True,
                                                                             method='Boosting', optimize=optimize_metric,
                                                                             verbose=verbose_bool)
                                        estimator_names.append('boosted_across')
                                        estimators_list.append(boosted_across_model)

                                elif ensemble_across_method == 'Blending':
                                    for across_etimators in enumerate(across_etimators_set):
                                        blend_across_model = blend_models(across_etimators, fold=k_fold,
                                                                          choose_better=True,
                                                                          optimize=optimize_metric,
                                                                          verbose=verbose_bool)
                                        estimator_names.append('blend_across')
                                        estimators_list.append(blend_across_model)

                                elif ensemble_across_method == 'Stacking':
                                    for across_etimators in enumerate(across_etimators_set):
                                        stack_across_model = stack_models(across_etimators, fold=k_fold,
                                                                          choose_better=True,
                                                                          optimize=optimize_metric,
                                                                          verbose=verbose_bool)
                                        estimator_names.append('stack_across')
                                        estimators_list.append(stack_across_model)


                        '''
                        Execute Model prediction and convert results to RMSE
                        '''
                        # The first step is to perform the 'finalize_model()' step on the created model to retrain
                        # the model(with all of the seen data as input), thus making predictions on the unseen data set
                        unseen_rmse = []
                        train_acc_results = []
                        test_acc_results = []
                        for n, name, model in zip(range(len(estimator_names)), estimator_names, estimators_list):
                            train_acc, test_acc, rmse = get_metric_pre_rmse(species_obs_df, obs_doys, species_name,
                                                           species_file_path + name, filename_suffix, n, model,
                                                           data_unseen, 'data_unseen', data_unseen_date, years_unseen,
                                                           phenophase_fluctuation, judge_period_days, zero_ratio,
                                                        year_col, date_col, save_bool=save_bin_doy_bool)
                            unseen_rmse.append(rmse)
                            train_acc_results.append(train_acc)
                            test_acc_results.append(test_acc)

                        unseen_rmse_results = unseen_rmse_results.append(pd.DataFrame({'phenophase': phenophase,
                                                     'species': species_name,
                                                     'window_days': window_days,
                                                     'pos_sample_size': pos_sample_size,
                                                     'estimator': estimator_names,
                                                     'RMSE': unseen_rmse,
                                                     'train_acc_results': train_acc_results,
                                                     'test_acc_results': test_acc_results,
                                                     'weather_features': '|'.join(weather_features),
                                                     'model_input': '|'.join(model_input),
                                                     'seed': int(random_state)
                                                     }))

                        '''
                        Final Predict weather_years phenological data
                        '''
                        # First, all the year observations will be used for setup
                    if weather_year_prediction:

                        if select_best_for_prediction:
                            pre_models = [create_model(i, fold=k_fold, verbose=verbose_bool) for i in model_input]
                            unseen_rmse = []
                            train_acc_results = []
                            for n, model in enumerate(pre_models):
                                train_acc, test_acc, rmse = get_metric_pre_rmse(
                                    species_obs_df, obs_doys, species_name, species_file_path + 'best',
                                    filename_suffix, n, model, data_unseen, 'data_unseen', data_unseen_date,
                                    years_unseen, phenophase_fluctuation, judge_period_days, zero_ratio, year_col,
                                    date_col, save_bool=save_bin_doy_bool)

                                unseen_rmse.append(rmse)
                                train_acc_results.append(train_acc)

                            if best_model_metric == 'Accuracy':
                                model_best_str = model_input[train_acc_results.index(max(train_acc_results))]
                            elif best_model_metric == 'RMSE':
                                model_best_str = model_input[unseen_rmse.index(min(unseen_rmse))]

                            phe_clf_prediction = setup(data=data_known, target=bloom_col, train_size=train_ratio,
                                                       session_id=random_state, html=False, silent=True,
                                                       verbose=verbose_bool)

                            pre_model = create_model(model_best_str, fold=k_fold, verbose=verbose_bool)

                            if tune_pre:
                                tuned_pre_model = tune_model(pre_model, fold=k_fold, n_iter=tune_iteration,
                                                     choose_better=True, optimize=optimize_metric, verbose=verbose_bool)
                                final_pre_model = tuned_pre_model
                            else:
                                final_pre_model = pre_model

                            get_metric_pre_rmse(species_obs_df, obs_doys, species_name, species_file_path,
                                filename_suffix, 0, final_pre_model, data_prediction, 'prediction',
                                 data_prediction_date, weather_years[1:], phenophase_fluctuation,
                                 judge_period_days, zero_ratio, year_col, date_col,
                                save_bool=save_bin_doy_bool, RMSE_bool=False)

                        else:
                            pre_model = create_model(model_input[0], fold=k_fold, verbose=verbose_bool)

                            if tune_pre:
                                tuned_pre_model = tune_model(pre_model, fold=k_fold, n_iter=tune_iteration,
                                                             choose_better=True, optimize=optimize_metric,
                                                             verbose=verbose_bool)
                                final_pre_model = tuned_pre_model
                            else:
                                final_pre_model = pre_model

                            get_metric_pre_rmse(species_obs_df, obs_doys, species_name, species_file_path ,
                                             filename_suffix, 0, final_pre_model, data_prediction, 'prediction',
                                             data_prediction_date, weather_years[1:], phenophase_fluctuation, judge_period_days,
                                             zero_ratio, year_col, date_col, save_bool=True, RMSE_bool=False)

                        # Analyze the models
                        # for model in final_models:
                        #     model_type = str(type(model))
                        #     if plot_feature:
                        #         if model_type in models_to_plot:
                        #             plot_model(model, plot='feature', save=True, verbose=verbose_bool)
                        #             os.rename(plot_feature_name,
                        #                       'feature_importance-' + phenophase + '-' + species_name + '-' + 'window_days_' + str(
                        #                           window_days) + '-' + models_ID[model_type] + '.png')
                        #         else:
                        #             pass
                        #     elif interpret_summary:
                        #         if model_type in models_to_interpret:
                        #             interpret_model(model, plot='summary')



    # Save RMSE results
    if save_rmse:
        rmse_results_file_path = result_dir + '/' + rmse_file_name

        if os.path.exists(rmse_results_file_path):
            unseen_rmse_results = pd.read_csv(rmse_results_file_path).append(unseen_rmse_results).reset_index(drop=True)
            unseen_rmse_results.to_csv(rmse_results_file_path, index=False, encoding='utf_8_sig')
        else:
            unseen_rmse_results.to_csv(rmse_results_file_path, index=False, encoding='utf_8_sig')

    # Save compare_grid results
    if compare_baseline:
        compare_grid_results_file_path = result_dir + '/' + compare_grid_file_name

        if os.path.exists(compare_grid_results_file_path):
            compare_grid_results = pd.read_csv(compare_grid_results_file_path).append(compare_grid_results).reset_index(drop=True)
            compare_grid_results.to_csv(compare_grid_results_file_path, index=False, encoding='utf_8_sig')
        else:
            compare_grid_results.to_csv(compare_grid_results_file_path, index=False, encoding='utf_8_sig')
