import numpy as np

"""
The following definitions are base functions used in biological models whose file name is 'models_functions.py'.
"""

def r_c_1(T, T_bc):
    '''
    
    Parameters
    ----------
    T : float
        The mean of daily temperature on D_i.
    T_bc : float
        Base temperature for rate of chilling.

    Returns
    -------
    r_c : float
          Rate of chilling temperature on D_i.

    '''
    return 0 if T > T_bc else 1

def r_c_2(T, T_bc):
    '''
    
    Parameters
    ----------
    T : float
        The mean of daily temperature on D_i.
    T_bc : float
           Base temperature for rate of chilling.

    Returns
    -------
    r_c : float
          Rate of chilling temperature on D_i.

    '''
    return 1 / T_bc if T < T_bc else 1 / T

def r_c_3(T, T_o, T_l, T_u):
    '''
    
    Parameters
    ----------
    T : float
        The mean of daily temperature on D_i.
    T_o : float
          Optimum air temperature for rate of chilling.
    T_l : float
          Lower threshold air temperature for rate of chilling.
    T_u : float
          Upper threshold air temperature for rate of chilling.

    Returns
    -------
    r_c : float
          Rate of chilling temperature on D_i.

    '''
    values_sorted = sorted([T_o, T_l, T_u])
    T_o, T_l, T_u = values_sorted[1], values_sorted[0], values_sorted[2]
    if T_l < T <= T_o:
        r_c = (T - T_l) / (T_o - T_l)
    elif T_o < T <= T_u:
        r_c = (T - T_u) / (T_o - T_u)
    else:
        r_c = 0
    
    return r_c

def r_c_4(T, T_o):
    '''
    
    Parameters
    ----------
    T : float
        The mean of daily temperature on D_i.
    T_o : float
          Optimum air temperature for rate of chilling.

    Returns
    -------
    r_c : float
          Rate of chilling temperature on D_i.

    '''
    if -3.4 < T <= T_o:
        r_c = (T + 3.4) / (T_o + 3.4)
    elif T_o < T <= 10.4:
        r_c = (T - 10.4) / (T_o - 10.4)
    else:
        r_c = 0
    
    return r_c

def r_c_5(T, m, n, p):
    '''
    
    Parameters
    ----------
    T : float
        The mean of daily temperature on D_i.
    m : float
        The constant in base function.
    n : float
        The constant in base function.
    p : float
        The constant in base function.

    Returns
    -------
    r_c : float
          Rate of chilling temperature on D_i.

    '''
    return 1 / (1 + np.exp(min(m * np.power(T - p, 2) + n * (T - p), 709.7827)))

def r_d_1(T, q, r):
    '''
    
    Parameters
    ----------
    T : float
        The mean of daily temperature on D_i.
    q : float
        The constant in base function.
    r : float
        The constant in base function.

    Returns
    -------
    r_d : float
          Rate of dormancy induction temperature on D_i.

    '''
    return 1 / (1 + np.exp(min(q * (T - r), 709.7827)))

def r_f_1(T):
    '''
    
    Parameters
    ----------
    T : float
        The mean of daily temperature on D_i.
    
    Returns
    -------
    r_f : float
          Rate of forcing temperature on D_i.
    
    '''
    return T

def r_f_2(T):
    '''
    
    Parameters
    ----------
    T : float
        The mean of daily temperature on D_i.
    
    Returns
    -------
    r_f : float
          Rate of forcing temperature on D_i.
    
    '''
    return 0 if T <= 0 else T

def r_f_3(T):
    '''
    
    Parameters
    ----------
    T : float
        The mean of daily temperature on D_i.
    
    Returns
    -------
    r_f : float
          Rate of forcing temperature on D_i.
    
    '''
    return 0 if T <= 0 else np.power(T, 2)

def r_f_4(T, T_bf):
    '''
    
    Parameters
    ----------
    T : float
        The mean of daily temperature on D_i.
    T_bf : float
           Base temperature for rate of forcing.
    
    Returns
    -------
    r_f : float
          Rate of forcing temperature on D_i.
    
    '''
    return 0 if T <= T_bf else T - T_bf

def r_f_5(T, a, b, c):
    '''
    
    Parameters
    ----------
    T : float
        The mean of daily temperature on D_i.
    a : float
        The constant in base function.
    b : float
        The constant in base function.
    c : float
        The constant in base function.
    
    Returns
    -------
    r_f : float
          Rate of forcing temperature on D_i.
    
    '''
    return a + b * np.exp(min(c * T, 709.7827))

def r_f_6(T, T_bf, d, e):
    '''
    
    Parameters
    ----------
    T : float
        The mean of daily temperature on D_i.
    T_bf : float
           Base temperature for rate of forcing.
    d : float
        The constant in base function.
    e : float
        The constant in base function.
    
    Returns
    -------
    r_f : float
          Rate of forcing temperature on D_i.
    
    '''
    return 0 if T <= T_bf else 1 / (1 + np.exp(min(d * (T - e), 709.7827)))

def r_f_7(T):
    '''
    
    Parameters
    ----------
    T : float
        The mean of daily temperature on D_i.
    
    Returns
    -------
    r_f : float
          Rate of forcing temperature on D_i.
    
    '''
    return 0 if T <= 0 else 28.4 / (1 + np.exp(min(-0.185 * (T - 18.4), 709.7827)))

def r_f_8(T_min, T_max, T_bf):
    '''
    
    Parameters
    ----------
    T_min : float
            The maximum temperature on D_i.
    T_max : float
            The minimum temperature on D_i.
    T_bf : float
           Base temperature for rate of forcing.

    Returns
    -------
    r_f : float
          Rate of forcing temperature on D_i.

    '''
    return 0 if T_max <= T_bf else np.power(T_max - T_bf, 2) / (T_max - T_min)

def r_f_9(T, T_min, T_max, T_bf):
    '''
    
    Parameters
    ----------
    T : float
        The mean of daily temperature on D_i.
    T_min : float
            The maximum temperature on D_i.
    T_max : float
            The minimum temperature on D_i.
    T_bf : float
           Base temperature for rate of forcing.

    Returns
    -------
    r_f : float
          Rate of forcing temperature on D_i.

    '''
    return 0 if T <= T_bf else T_min + 0.65 * (T_max - T_min) - T_bf

def r_f_10(T, L, S_c, C_crit, g, h, f):
    '''
    
    Parameters
    ----------
    T : float
        The mean of daily temperature on D_i.
    L : float
        Day length of D_i.
    S_c : float
          State of chilling, that is the integral of rate of chilling.
    C_crit : float
             Critical value of state of chilling for the transition from rest to quiescence.
    g : float
        The constant in base function.
    h : float
        The constant in base function.
    f : float
        The constant in base function.

    Returns
    -------
    r_f : float
          Rate of forcing temperature on D_i.

    '''
    # DL_50 : Corresponds to the critical daylength at which the temperature of mid-response of the rate of forcing, R_f(D_i), is 30°C.
    DL_50 = 24 / (1 + np.exp(min(h * (S_c - C_crit), 709.7827)))
    # t_50 : The temperature of mid-response.
    t_50 = 60 / (1 + np.exp(min(g * (L - DL_50), 709.7827)))
    r_f = 1 / (1 + np.exp(min(-f * (T - t_50), 709.7827)))
    
    return r_f

def r_l_1(L):
    '''
    
    Parameters
    ----------
    L : float
        Day length of D_i.

    Returns
    -------
    r_l : float
          Rate of photoperiod on D_i.

    '''
    return L

def r_l_2():
    '''
    
    Returns
    -------
    r_l : float
          Rate of photoperiod on D_i.

    '''
    return 12

def r_l_3(L):
    '''
    
    Parameters
    ----------
    L : float
        Day length of D_i.

    Returns
    -------
    r_l : float
          Rate of photoperiod on D_i.

    '''
    return L / 24

def r_l_4(L, k):
    '''
    
    Parameters
    ----------
    L : float
        Day length of D_i.
    k : float
        The constant in base function.

    Returns
    -------
    r_l : float
          Rate of photoperiod on D_i.

    '''
    return np.power(L / 10, k)

def r_l_5(L, L_crit):
    '''
    
    Parameters
    ----------
    L : float
        Day length of D_i.
    L_crit : float
             Critical value of photoperiod for dormancy induction.

    Returns
    -------
    r_l : float
          Rate of photoperiod on D_i.

    '''
    return 1 / (1 + np.exp(min(10 * (L - L_crit), 709.7827)))

def K_1(S_c, C_crit):
    '''
    
    Parameters
    ----------
    S_c : float
          State of chilling, that is the integral of rate of chilling.
    C_crit : float
             Critical value of state of chilling for the transition from rest to quiescence.

    Returns
    -------
    K : float
        Growth competence: bud's potential to respond to air temperature [0, 1].

    '''
    return 0 if S_c < C_crit else 1

def K_2(S_c, C_crit, K_min):
    '''
    
    Parameters
    ----------
    S_c : float
          State of chilling, that is the integral of rate of chilling.
    C_crit : float
             Critical value of state of chilling for the transition from rest to quiescence.
    K_min : float
            Minimum potential of unchilled bud to respond to forcing temperature.

    Returns
    -------
    K : float
        Growth competence: bud's potential to respond to air temperature [0, 1].

    '''
    return K_min + (1 - K_min) / C_crit * S_c if S_c < C_crit else 1

def K_3(S_c, C_crit, C_dr, K_min):
    '''
    
    Parameters
    ----------
    S_c : float
          State of chilling, that is the integral of rate of chilling.
    C_crit : float
             Critical value of state of chilling for the transition from rest to quiescence.
    C_dr : float
           Critical state of chilling for transition from deepening rest to decreasing rest.
    K_min : float
            Minimum potential of unchilled bud to respond to forcing temperature.

    Returns
    -------
    K : float
        Growth competence: bud's potential to respond to air temperature [0, 1].

    '''
    values_sorted = sorted([C_dr, C_crit])
    C_dr, C_crit = values_sorted[0], values_sorted[1]
    if S_c < C_dr:
        K = 1 - (1 - K_min) * S_c / C_dr
    elif C_dr <= S_c < C_crit:
        K = K_min + (1 - K_min) * (S_c - C_dr) / (C_crit - C_dr)
    else:
        K = 1
    
    return K

def K_4(T, S_c, C_crit, C_pr, C_tr, T_1, T_2):
    '''
    
    Parameters
    ----------
    T : float
        The mean of daily temperature on D_i.
    S_c : float
          State of chilling, that is the integral of rate of chilling.
    C_crit : float
             Critical value of state of chilling for the transition from rest to quiescence.
    C_pr : float
           Critical value of state of chilling for transition from true rest to post-rest.
    C_tr : float
           Critical value of state of chilling for transition from pre-rest to true rest.
    T_1 : float
          Lower value of temperature range for which development is possible.
    T_2 : float
          Upper value of temperature range for which development is possible.

    Returns
    -------
    K : float
        Growth competence: bud's potential to respond to air temperature [0, 1].

    '''
    values_sorted = sorted([C_pr, C_crit])
    C_pr, C_crit = values_sorted[0], values_sorted[1]
    # T_trh : Temperature threshold above which development is possible and below which development is impossible
    if S_c < C_tr:
        T_trh = T_1 + (T_2 - T_1) / C_tr * S_c
    elif C_pr <= S_c < C_crit:
        T_trh = T_1 + (T_1 - T_2) * (S_c - C_crit) / (C_crit - C_pr) * S_c
    else:
        T_trh = False
    
    if T_trh:
        if S_c < C_tr and T > T_trh:
            K = 1
        elif C_pr <= S_c < C_crit and T > T_trh:
            K = 1
        elif S_c >= C_crit:
            K = 1
        else:
            K = 0
    else:
        K = 1 if S_c >= C_crit else 0
    
    return K

def K_5(S_d, D_crit):
    '''
    
    Parameters
    ----------
    S_d : float
          State of dormancy induction, integral of rate of dormancy induction.
    D_crit : float
             Critical value of state of chilling for the transition from dormancy induction to rest.

    Returns
    -------
    K : float
        Growth competence: bud's potential to respond to air temperature [0, 1].

    '''
    return 0 if S_d < D_crit else 1




