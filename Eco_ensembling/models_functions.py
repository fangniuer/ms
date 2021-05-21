from base_functions import *

import numpy as np

"""
The following definitions are functions of all biological models to be used by 'BioModels.construct_model' method in the file named 'bio_models.py'.
"""

def TC(
       t_s, F_crit,
       years, weather, flower, year_column, T_mean_column, flower_date_column,
       max_error, max_date, indicator='RMSE'
       ):
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts = list(weather[weather[year_column] == year][T_mean_column].values)
        R_fs = [r_f_1(T) for T in Ts]

        error = max_error
        y = max_date
        S_f = 0
        for t in range(int(round(t_s)) - 1, max_date):
            S_f += R_fs[t]

            if S_f >= F_crit:
                y = t + 1
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 2
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def RI(
       t_s, F_crit,
       years, weather, flower, year_column, T_mean_column, flower_date_column,
       max_error, max_date, indicator='RMSE'
       ):
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts = list(weather[weather[year_column] == year][T_mean_column].values)
        R_fs = [r_f_2(T) for T in Ts]

        error = max_error
        y = max_date
        S_f = 0
        for t in range(int(round(t_s)) - 1, max_date):
            S_f += R_fs[t]

            if S_f >= F_crit:
                y = t + 1
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 2
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def AMC(
        t_s, F_crit,
        years, weather, flower, year_column, T_mean_column, flower_date_column,
        max_error, max_date, indicator='RMSE'
        ):
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts = list(weather[weather[year_column] == year][T_mean_column].values)
        R_fs = [r_f_3(T) for T in Ts]

        error = max_error
        y = max_date
        S_f = 0
        for t in range(int(round(t_s)) - 1, max_date):
            S_f += R_fs[t]

            if S_f >= F_crit:
                y = t + 1
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 2
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def GDD(
        t_s, F_crit, T_bf,
        years, weather, flower, year_column, T_mean_column, flower_date_column,
        max_error, max_date, indicator='RMSE'
        ):
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts = list(weather[weather[year_column] == year][T_mean_column].values)
        R_fs = [r_f_4(T, T_bf) for T in Ts]

        error = max_error
        y = max_date
        S_f = 0
        for t in range(int(round(t_s)) - 1, max_date):
            S_f += R_fs[t]

            if S_f >= F_crit:
                y = t + 1
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 3
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def EX(
       t_s, F_crit, a, b, c,
       years, weather, flower, year_column, T_mean_column, flower_date_column,
       max_error, max_date, indicator='RMSE'
       ):
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts = list(weather[weather[year_column] == year][T_mean_column].values)
        R_fs = [r_f_5(T, a, b, c) for T in Ts]

        error = max_error
        y = max_date
        S_f = 0
        for t in range(int(round(t_s)) - 1, max_date):
            S_f += R_fs[t]

            if S_f >= F_crit:
                y = t + 1
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 5
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def UF(
       t_s, F_crit, T_bf, d, e,
       years, weather, flower, year_column, T_mean_column, flower_date_column,
       max_error, max_date, indicator='RMSE'
       ):
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts = list(weather[weather[year_column] == year][T_mean_column].values)
        R_fs = [r_f_6(T, T_bf, d, e) for T in Ts]

        error = max_error
        y = max_date
        S_f = 0
        for t in range(int(round(t_s)) - 1, max_date):
            S_f += R_fs[t]

            if S_f >= F_crit:
                y = t + 1
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 5
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def FU(
       t_s, F_crit,
       years, weather, flower, year_column, T_mean_column, flower_date_column,
       max_error, max_date, indicator='RMSE'
       ):
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts = list(weather[weather[year_column] == year][T_mean_column].values)
        R_fs = [r_f_7(T) for T in Ts]

        error = max_error
        y = max_date
        S_f = 0
        for t in range(int(round(t_s)) - 1, max_date):
            S_f += R_fs[t]

            if S_f >= F_crit:
                y = t + 1
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 2
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def PU(
       t_s, F_crit, T_bf,
       years, weather, flower, year_column, T_mean_column, L_column, flower_date_column,
       max_error, max_date, indicator='RMSE'
       ):
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts = list(weather[weather[year_column] == year][T_mean_column].values)
        Ls = list(weather[weather[year_column] == year][L_column].values)
        r_fs = [r_f_4(T, T_bf) for T in Ts]
        r_ls = [r_l_1(L) for L in Ls]
        R_fs = list(np.array(r_fs) * np.array(r_ls))

        error = max_error
        y = max_date
        S_f = 0
        for t in range(int(round(t_s)) - 1, max_date):
            S_f += R_fs[t]

            if S_f >= F_crit:
                y = t + 1
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 3
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def DH(
       t_s, F_crit, T_bf,
       years, weather, flower, year_column, T_min_column, T_max_column, flower_date_column,
       max_error, max_date, indicator='RMSE'
       ):
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        T_mins = list(weather[weather[year_column] == year][T_min_column].values)
        T_maxs = list(weather[weather[year_column] == year][T_max_column].values)
        R_fs = [r_f_8(T_mins[i], T_maxs[i], T_bf) * r_l_2() for i in range(len(T_mins))]
        
        error = max_error
        y = max_date
        S_f = 0
        for t in range(int(round(t_s)) - 1, max_date):
            S_f += R_fs[t]

            if S_f >= F_crit:
                y = t + 1
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 3
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def PT(
       t_s, F_crit, T_bf,
       years, weather, flower, year_column, T_mean_column, T_min_column, T_max_column, L_column, flower_date_column,
       max_error, max_date, indicator='RMSE'
       ):
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts = list(weather[weather[year_column] == year][T_mean_column].values)
        T_mins = list(weather[weather[year_column] == year][T_min_column].values)
        T_maxs = list(weather[weather[year_column] == year][T_max_column].values)
        Ls = list(weather[weather[year_column] == year][L_column].values)
        R_fs = [r_f_9(Ts[i], T_mins[i], T_maxs[i], T_bf) * r_l_3(Ls[i]) for i in range(len(T_mins))]
        
        error = max_error
        y = max_date
        S_f = 0
        for t in range(int(round(t_s)) - 1, max_date):
            S_f += R_fs[t]

            if S_f >= F_crit:
                y = t + 1
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 3
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def M1(
       t_s, F_crit, T_bf, k,
       years, weather, flower, year_column, T_mean_column, L_column, flower_date_column,
       max_error, max_date, indicator='RMSE'
       ):
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts = list(weather[weather[year_column] == year][T_mean_column].values)
        Ls = list(weather[weather[year_column] == year][L_column].values)
        r_fs = [r_f_4(T, T_bf) for T in Ts]
        r_ls = [r_l_4(L, k) for L in Ls]
        R_fs = list(np.array(r_fs) * np.array(r_ls))

        error = max_error
        y = max_date
        S_f = 0
        for t in range(int(round(t_s)) - 1, max_date):
            S_f += R_fs[t]

            if S_f >= F_crit:
                y = t + 1
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 4
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def SE(
       F_crit, T_bf, t_s, T_o, C_crit, d, e,
       years, weather, flower, year_column, T_mean_column, flower_date_column,
       max_error, max_date, last_year_start_date_neg, indicator='RMSE'
       ):
    
    t_s = t_s + 1 - last_year_start_date_neg
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts_last_year = list(weather[weather[year_column] == year - 1][T_mean_column].values)[last_year_start_date_neg:]
        Ts_this_year = list(weather[weather[year_column] == year][T_mean_column].values)
        Ts = Ts_last_year + Ts_this_year
        r_fs = [r_f_6(T, T_bf, d, e) for T in Ts]
        R_cs = [r_c_4(T, T_o) for T in Ts]
        
        error = max_error
        y = max_date
        S_f = 0
        S_c = 0
        for t in range(int(round(t_s)) - 1, max_date - last_year_start_date_neg):
            S_c += R_cs[t]
            K = K_1(S_c, C_crit)
            S_f += K * r_fs[t]
            
            if S_f >= F_crit:
                y = t + 1 - 1 + last_year_start_date_neg
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 7
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def UC(
       F_crit, T_bf, t_s, C_crit, d, e, m, n, p,
       years, weather, flower, year_column, T_mean_column, flower_date_column,
       max_error, max_date, last_year_start_date_neg, indicator='RMSE'
       ):
    
    t_s = t_s + 1 - last_year_start_date_neg
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts_last_year = list(weather[weather[year_column] == year - 1][T_mean_column].values)[last_year_start_date_neg:]
        Ts_this_year = list(weather[weather[year_column] == year][T_mean_column].values)
        Ts = Ts_last_year + Ts_this_year
        r_fs = [r_f_6(T, T_bf, d, e) for T in Ts]
        R_cs = [r_c_5(T, m, n, p) for T in Ts]
        
        error = max_error
        y = max_date
        S_f = 0
        S_c = 0
        for t in range(int(round(t_s)) - 1, max_date - last_year_start_date_neg):
            S_c += R_cs[t]
            K = K_1(S_c, C_crit)
            S_f += K * r_fs[t]
            
            if S_f >= F_crit:
                y = t + 1 - 1 + last_year_start_date_neg
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 9
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def AL(
       T_bf, T_bc, t_s, C_crit, alpha, beta, gamma,
       years, weather, flower, year_column, T_mean_column, flower_date_column,
       max_error, max_date, last_year_start_date_neg, indicator='RMSE'
       ):
    
    t_s = t_s + 1 - last_year_start_date_neg
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts_last_year = list(weather[weather[year_column] == year - 1][T_mean_column].values)[last_year_start_date_neg:]
        Ts_this_year = list(weather[weather[year_column] == year][T_mean_column].values)
        Ts = Ts_last_year + Ts_this_year
        r_fs = [r_f_4(T, T_bf) for T in Ts]
        R_cs = [r_c_1(T, T_bc) for T in Ts]
        
        error = max_error
        y = max_date
        S_f = 0
        S_c = 0
        for t in range(int(round(t_s)) - 1, max_date - last_year_start_date_neg):
            S_c += R_cs[t]
            K = K_1(S_c, C_crit)
            S_f += K * r_fs[t]
            F_crit = alpha + beta * np.exp(min(gamma * S_c, 702.8749))
            
            if S_f >= F_crit:
                y = t + 1 - 1 + last_year_start_date_neg
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 7
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def UM(
       T_bf, t_s, C_crit, d, e, m, n, p, omega, epsilon,
       years, weather, flower, year_column, T_mean_column, flower_date_column,
       max_error, max_date, last_year_start_date_neg, indicator='RMSE'
       ):
    
    t_s = t_s + 1 - last_year_start_date_neg
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts_last_year = list(weather[weather[year_column] == year - 1][T_mean_column].values)[last_year_start_date_neg:]
        Ts_this_year = list(weather[weather[year_column] == year][T_mean_column].values)
        Ts = Ts_last_year + Ts_this_year
        r_fs = [r_f_6(T, T_bf, d, e) for T in Ts]
        R_cs = [r_c_5(T, m, n, p) for T in Ts]
        
        error = max_error
        y = max_date
        S_f = 0
        S_c = 0
        for t in range(int(round(t_s)) - 1, max_date - last_year_start_date_neg):
            S_c += R_cs[t]
            K = K_1(S_c, C_crit)
            S_f += K * r_fs[t]
            F_crit = omega * np.exp(min(epsilon * S_c, 702.8749))
            
            if S_f >= F_crit:
                y = t + 1 - 1 + last_year_start_date_neg
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 10
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def PA(
       F_crit, t_s, C_crit, T_o, K_min,
       years, weather, flower, year_column, T_mean_column, flower_date_column,
       max_error, max_date, last_year_start_date_neg, indicator='RMSE'
       ):
    
    t_s = t_s + 1 - last_year_start_date_neg
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts_last_year = list(weather[weather[year_column] == year - 1][T_mean_column].values)[last_year_start_date_neg:]
        Ts_this_year = list(weather[weather[year_column] == year][T_mean_column].values)
        Ts = Ts_last_year + Ts_this_year
        r_fs = [r_f_7(T) for T in Ts]
        R_cs = [r_c_4(T, T_o) for T in Ts]
        
        error = max_error
        y = max_date
        S_f = 0
        S_c = 0
        for t in range(int(round(t_s)) - 1, max_date - last_year_start_date_neg):
            S_c += R_cs[t]
            K = K_2(S_c, C_crit, K_min)
            S_f += K * r_fs[t]
            
            if S_f >= F_crit:
                y = t + 1 - 1 + last_year_start_date_neg
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 5
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def GU(
       F_crit, t_s, C_crit, T_bc, T_bf, K_min,
       years, weather, flower, year_column, T_mean_column, flower_date_column,
       max_error, max_date, last_year_start_date_neg, indicator='RMSE'
       ):
    
    t_s = t_s + 1 - last_year_start_date_neg
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts_last_year = list(weather[weather[year_column] == year - 1][T_mean_column].values)[last_year_start_date_neg:]
        Ts_this_year = list(weather[weather[year_column] == year][T_mean_column].values)
        Ts = Ts_last_year + Ts_this_year
        r_fs = [r_f_4(T, T_bf) for T in Ts]
        R_cs = [r_c_2(T, T_bc) for T in Ts]
        
        error = max_error
        y = max_date
        S_f = 0
        S_c = 0
        for t in range(int(round(t_s)) - 1, max_date - last_year_start_date_neg):
            S_c += R_cs[t]
            K = K_2(S_c, C_crit, K_min)
            S_f += K * r_fs[t]
            
            if S_f >= F_crit:
                y = t + 1 - 1 + last_year_start_date_neg
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 6
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def DOR(
        F_crit, t_s, C_crit, D_crit, L_crit, f, g, h, m, n, p, q, r,
        years, weather, flower, year_column, T_mean_column, L_column, flower_date_column,
        max_error, max_date, last_year_start_date_neg, indicator='RMSE'
        ):
    
    t_s = t_s + 1 - last_year_start_date_neg
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts_last_year = list(weather[weather[year_column] == year - 1][T_mean_column].values)[last_year_start_date_neg:]
        Ts_this_year = list(weather[weather[year_column] == year][T_mean_column].values)
        Ts = Ts_last_year + Ts_this_year
        Ls_last_year = list(weather[weather[year_column] == year - 1][L_column].values)[last_year_start_date_neg:]
        Ls_this_year = list(weather[weather[year_column] == year][L_column].values)
        Ls = Ls_last_year + Ls_this_year
        r_ds = [r_d_1(T, q, r) for T in Ts]
        r_ls = [r_l_5(L, L_crit) for L in Ls]
        r_cs = [r_c_5(T, m, n, p) for T in Ts]
        R_ds = list(np.array(r_ds) * np.array(r_ls))
        
        error = max_error
        y = max_date
        S_f = 0
        S_c = 0
        S_d = 0
        for t in range(int(round(t_s)) - 1, max_date - last_year_start_date_neg):
            S_d += R_ds[t]
            K = K_5(S_d, D_crit)
            S_c += K * r_cs[t]
            r_f = r_f_10(Ts[t], Ls[t], S_c, C_crit, g, h, f)
            S_f += K * r_f
            
            if S_f >= F_crit and S_d >= D_crit:
                y = t + 1 - 1 + last_year_start_date_neg
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 13
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def DR(
       F_crit, T_bf, t_s, T_o, T_l, T_u, C_crit, C_dr, K_min, d, e,
       years, weather, flower, year_column, T_mean_column, flower_date_column,
       max_error, max_date, last_year_start_date_neg, indicator='RMSE'
       ):
    
    t_s = t_s + 1 - last_year_start_date_neg
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts_last_year = list(weather[weather[year_column] == year - 1][T_mean_column].values)[last_year_start_date_neg:]
        Ts_this_year = list(weather[weather[year_column] == year][T_mean_column].values)
        Ts = Ts_last_year + Ts_this_year
        r_fs = [r_f_6(T, T_bf, d, e) for T in Ts]
        R_cs = [r_c_3(T, T_o, T_l, T_u) for T in Ts]
        
        error = max_error
        y = max_date
        S_f = 0
        S_c = 0
        for t in range(int(round(t_s)) - 1, max_date - last_year_start_date_neg):
            S_c += R_cs[t]
            K = K_3(S_c, C_crit, C_dr, K_min)
            S_f += K * r_fs[t]
            
            if S_f >= F_crit:
                y = t + 1 - 1 + last_year_start_date_neg
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 11
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")

def FP(
       F_crit, T_bf, t_s, T_o, T_l, T_u, C_crit, C_tr, C_pr, T_1, T_2, d, e,
       years, weather, flower, year_column, T_mean_column, flower_date_column,
       max_error, max_date, last_year_start_date_neg, indicator='RMSE'
       ):
    
    t_s = t_s + 1 - last_year_start_date_neg
    
    ys = []
    errors = []
    obss = []
    
    for year in years:
        Ts_last_year = list(weather[weather[year_column] == year - 1][T_mean_column].values)[last_year_start_date_neg:]
        Ts_this_year = list(weather[weather[year_column] == year][T_mean_column].values)
        Ts = Ts_last_year + Ts_this_year
        r_fs = [r_f_6(T, T_bf, d, e) for T in Ts]
        R_cs = [r_c_3(T, T_o, T_l, T_u) for T in Ts]
        
        error = max_error
        y = max_date
        S_f = 0
        S_c = 0
        for t in range(int(round(t_s)) - 1, max_date - last_year_start_date_neg):
            S_c += R_cs[t]
            K = K_4(Ts[t], S_c, C_crit, C_pr, C_tr, T_1, T_2)
            S_f += K * r_fs[t]
            
            if S_f >= F_crit:
                y = t + 1 - 1 + last_year_start_date_neg
                break
        
        if indicator == 'ys':
            ys.append(y)
        elif indicator == 'RMSE' or 'AIC&NSE':
            obs = flower[flower[year_column] == year][flower_date_column].values[0]
            error = obs - y
            obss.append(obs)
            errors.append(error)
        else:
            pass
    
    if indicator == 'ys':
        return ys
    elif indicator == 'RMSE':
        return np.sqrt(np.mean(np.power(errors, 2)))
    elif indicator == 'AIC&NSE':
        num_parameters = 13
        num_errors = len(errors)
        quadratic_sum_errors = np.sum(np.power(errors, 2))
        AIC = num_errors * np.log(quadratic_sum_errors / num_errors) + 2 * (num_parameters + 1)
        NSE = 1 - quadratic_sum_errors / np.sum(np.power(np.array(obss) - np.mean(obss), 2))
        return AIC, NSE
    else:
        raise ValueError("Unsupported [indicator]: '" + indicator + "'. The indicator should be 'ys', 'RMSE' or 'AIC&NSE'")
