"""
This module contains algorithms for identifying/removing/predicting outliers.
"""
import warnings
from functools import reduce
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pmdarima import ARIMA, AutoARIMA
from scipy import signal

from ..utility import NonnegativeInt, PositiveFlt, PositiveInt


def find_poly_time_trend(params: pd.Series, resid: np.ndarray, trend_offset: PositiveInt):
    """ 
    Find fitted polynomial time trend.
    
    :param params: The parameter from time series model.
    :param resid: Residual time series.
    :return: Fitted polynomial time trend which is equal in size of input endogenous time series.
    """
    time_obs = np.arange(start=trend_offset, stop=trend_offset+len(resid))

    other_poly = params[params.index.str.match(r'trend\.\d+')]
    if 'intercept' in params.index:
        other_poly['trend.0'] = params['intercept']
    else:
        other_poly['trend.0'] = 0
    if 'drift' in params.index:
        other_poly['trend.1'] = params['drift']
    else:
        other_poly['trend.1'] = 0
    other_poly.index = other_poly.index.str.replace(r'trend\.', '').astype(int)

    max_other_poly = other_poly.max()

    fitted_trend = np.zeros(time_obs.shape)
    for i in range(max_other_poly):
        if i in other_poly.index:
            fitted_trend += other_poly[i] * np.power(time_obs, i)

    return fitted_trend

def sarima_params_to_poly_coeffs(params: pd.Series,
                                 order: Tuple[NonnegativeInt, NonnegativeInt, NonnegativeInt, NonnegativeInt],
                                 seasonal_order: Tuple[NonnegativeInt, NonnegativeInt, NonnegativeInt, NonnegativeInt, NonnegativeInt]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """ 
    Convert coefficients to polynomial cofficients for AR, MA, sAR, sMA parameters, in decreasing polynomial orders.

    :param params: The parameter from time series model.
    :param order: The order of fitted model.
    :param seasonal_order: The seasonal order of fitted model.
    :return: Converted polynomials for AR polynomial, MA polynomial, sAR polynomial, sMA polynomial.
    """
    def fill_poly_with_zero(params):
        for i in range(params.index.max()):
            if i not in params:
                params[i] = 0

    difference_d = order[1]
    difference_seasonal_d = seasonal_order[1]
    seasonal_s = max(seasonal_order[3], 1)

    ar_params = params[params.index.str.match(r'ar\.L\d+')]
    ar_params *= -1
    ar_params.index = ar_params.index.str.replace(r'ar\.L', '', regex=True).astype(int)
    ar_params[0] = 1
    fill_poly_with_zero(ar_params)
    ar_params = ar_params.sort_index(ascending=False) ## Decreasing order in polynomial

    ar_seasonal_params = params[params.index.str.match(r'ar\.S\.L\d+')]
    ar_seasonal_params *= -1
    ar_seasonal_params.index = ar_seasonal_params.index.str.replace(r'ar\.S\.L', '', regex=True).astype(int)
    ar_seasonal_params[0] = 1
    fill_poly_with_zero(ar_seasonal_params)
    ar_seasonal_params = ar_seasonal_params.sort_index(ascending=False)

    diff_params = reduce(np.polymul, [np.array([-1, 1])] * difference_d, np.array([1]))
    diff_params = pd.Series(diff_params, index=np.flip(np.arange(len(diff_params))), name='coef')

    ma_params = params[params.index.str.match(r'ma\.L\d+')]
    ma_params.index = ma_params.index.str.replace(r'ma\.L', '', regex=True).astype(int)
    ma_params[0] = 1
    fill_poly_with_zero(ma_params)
    ma_params = ma_params.sort_index(ascending=False) ## Decreasing order in polynomial

    ma_seasonal_params = params[params.index.str.match(r'ma\.S\.L\d+')]
    ma_seasonal_params.index = ma_seasonal_params.index.str.replace(r'ma\.S\.L', '', regex=True).astype(int)
    ma_seasonal_params[0] = 1
    fill_poly_with_zero(ma_seasonal_params)
    ma_seasonal_params = ma_seasonal_params.sort_index(ascending=False)

    diff_seasonal_params = reduce(np.polymul, [np.concatenate((np.array([-1]), np.zeros(seasonal_s-1), np.array([1])))] * difference_seasonal_d, np.array([1]))
    diff_seasonal_params = pd.Series(diff_seasonal_params, index=np.flip(np.arange(len(diff_seasonal_params))), name='coef')

    return ar_params, diff_params, ma_params, ar_seasonal_params, diff_seasonal_params, ma_seasonal_params

def find_intercept(
        params: pd.Series,
        resid: np.ndarray,
        order: Tuple[NonnegativeInt, NonnegativeInt, NonnegativeInt, NonnegativeInt],
        seasonal_order: Tuple[NonnegativeInt, NonnegativeInt, NonnegativeInt, NonnegativeInt, NonnegativeInt],
        trend_offset: PositiveInt) -> np.ndarray:
    """
    Convert fitted trend to infinite MA representation. Can remove this value from original time series, the residuals is fitted by SARIMAX model.

    :param params: The parameter from time series model.
    :param resid: Residual time series.
    :param order: The order of fitted model.
    :param seasonal_order: The seasonal order of fitted model.
    :return: The fitted intercept with equal length to response.
    """
    ar_params, _, diff_params, ar_seasonal_params, diff_seasonal_params, _ = sarima_params_to_poly_coeffs(params, order, seasonal_order)

    left_params = reduce(np.polymul, [ar_params, diff_params, ar_seasonal_params, diff_seasonal_params])

    time_trend = find_poly_time_trend(params, resid, trend_offset)
    return time_trend / left_params.sum()

def params_to_infinite_representations(
        ar_params: pd.Series,
        diff_params: pd.Series,
        ma_params: pd.Series,
        ar_seasonal_params: pd.Series,
        diff_seasonal_params: pd.Series,
        ma_seasonal_params: pd.Series,
        leads: PositiveInt=100,
        right_on_left: bool=True) -> np.ndarray:
    """
    Convert fitted parameters to infinite MA representation.

    :param ar_params: Output from sarima_params_to_poly_coeffs.
    :param diff_params: Output from sarima_params_to_poly_coeffs.
    :param ma_params: Output from sarima_params_to_poly_coeffs.
    :param ar_params: Output from sarima_params_to_poly_coeffs.
    :param ar_seasonal_params: Output from sarima_params_to_poly_coeffs.
    :param diff_seasonal_params: Output from sarima_params_to_poly_coeffs.
    :param ma_seasonal_params: Output from sarima_params_to_poly_coeffs.
    :param leads: Truncate this number of polynomials to represent infinite ma representations.
    :return: Infinite MA(right_on_left) or AR(left_on_right) series in increasing polynomials.
    """
    left_params = reduce(np.polymul, [ar_params, diff_params, ar_seasonal_params, diff_seasonal_params])
    left_params = np.flip(left_params) ## Convert to increasing polynomials

    right_params = reduce(np.polymul, [ma_params, ma_seasonal_params])
    right_params = np.flip(right_params) ## Convert to increasing polynomials

    impulse = np.zeros(leads)
    impulse[0] = 1
    if right_on_left:
        return signal.lfilter(right_params, left_params, impulse)
    else:
        return signal.lfilter(left_params, right_params, impulse)

def recursive_filter(signal_x: np.ndarray, filter_param: np.ndarray, init=None):
    """
    Autoregressive, or recursive, filtering.

    :param signal_x: time series data.
    :param filter_param: AR coefficients in increasing time order.
    :param init : Initial values of the time series prior to the first value of y. The default is zero.
    :return: Filtered array, number of columns determined by x and filter. If x is a pandas object than a Series is returned.
    """
    if init is not None:  # integer init are treated differently in lfiltic
        assert init.shape == filter_param.shape, 'filter_param must be the same length as init.'

    if init is not None:
        signal_zi = signal.lfiltic([1], np.r_[1, -filter_param], init, signal_x)
    else:
        signal_zi = None

    signal_y = signal.lfilter([1.], np.r_[1, -filter_param], signal_x, zi=signal_zi)

    if init is not None:
        result = signal_y[0]
    else:
        result = signal_y

    return result

def diffinv(signal_x: np.ndarray, lag: PositiveInt=1, init: Optional[PositiveInt]=None):
    """
    Discrete Integration: Inverse of Differencing
    :param signal_x: A numeric 1d array.
    :param lag:	A scalar lag parameter.
    :param differences: An integer representing the order of the difference.
    :param init: A numeric vector, matrix, or time series containing the initial values for the integrals. If missing, zeros are used.
    :return: A numeric vector representing the discrete integral of x.
    """
    assert len(signal_x.shape) == 1, "x must be 1d array."
    if init is None:
        init = np.zeros(lag)
    assert init.shape == (lag,), 'init must be 1d array of length lag.'

    signal_x = np.concatenate((init, signal_x))

    all_arrays = []
    for i in range(lag):
        temp = np.full(len(signal_x) // lag + 1, np.nan)
        temp2 = np.cumsum(signal_x[i::lag])
        temp[:len(temp2)] = temp2
        all_arrays.append(temp)

    result = np.stack(all_arrays).transpose().flatten()
    result = result[~np.isnan(result)]
    return result

# def variance_change_on_residuals(
#         params: pd.Series,
#         resid: np.ndarray,
#         order: Tuple(NonnegativeInt, NonnegativeInt, NonnegativeInt, NonnegativeInt),
#         seasonal_order: Tuple(NonnegativeInt, NonnegativeInt, NonnegativeInt, NonnegativeInt, NonnegativeInt),
#         located_ol: pd.DataFrame) -> np.ndarray:
#     """
#     Variance Change outlier effect on residuals.S

#     :param params:
#     :param resid:
#     :param order:
#     :param seasonal_order:
#     :param located_ol:
#     """
#     return

class IterativeTtestOutlierDetection:
    """
    This class implements the traditional iterative procedure for identifying and removeing outliers.
    Five types of outliers can be considered. By default: "AO" additive outliers, "LS" level shifts, and "TC" temporary changes are selected; "IO" innovative outliers and "SLS" seasonal level shifts can also be selected.
    The algroithm iterates around locating and removing outliers first for the original series and then for the adjusted series. The process stops if no additional outliers are found in the current iteration or if maxit iterations are reached.
    """

    def __init__(
        self,
        types: pd.DataFrame=pd.DataFrame({'type': ['IO', 'AO', 'TC']}),
        maxit: PositiveInt=1,
        maxit_iloop: PositiveInt=4,
        maxit_oloop: PositiveInt=4,
        cval: Optional[PositiveFlt]=None,
        cval_reduce: PositiveFlt=0.14286,
        discard_method: Literal['en-masse', 'bottom-up']='en_masse',
        discard_cval: Optional[PositiveFlt]=None,
        tsmethod: Literal["AutoARIMA", "ARIMA"]='AutoARIMA',
        args_tsmethod: Optional[Dict[str, Any]]=None) -> None:
        """
        Initializer and configuration for IterativeTtestOutlierDetection.

        :param cval: The critical value to determine the significance of each type of outlier. If no value is specified for argument cval a default value based on the sample size is used. Let nn be the number of observations. If $n \leq 50$, then cval is set equal to 3.0; If $n \geq 450$, then cval is set equal to 4.0; otherwise cval is set equal to $3 + 0.0025 * (n - 50)3 + 0.0025 * (n - 50)$.
        :param types: A character list indicating the type of outlier to be considered by the detection procedure: innovational outliers ("IO"), additive outliers ("AO"), level shifts ("LS"), temporary changes ("TC") and seasonal level shifts ("SLS"). If None is provided, then a list of 'AO', 'LS', 'TC' is used.
        :param maxit: The maximum number of iterations.
        :param maxit_iloop: The maximum number of iterations in the inner loop. See locate_outliers.
        :param maxit_oloop: The maximum number of iterations in the outer loop.
        :param cval_reduce: Factor by which cval is reduced if the procedure is run on the adjusted series, if $maxit > 1$, the new critical value is defined as $cval * (1 - cval.reduce)$.
        :param discard_method: The method used in the second stage of the procedure. See discard.outliers.
        :param discard_cval: The critical value to determine the significance of each type of outlier in the second stage of the procedure (discard outliers). By default, the same critical value is used in the first stage of the procedure (location of outliers) and in the second stage (discard outliers). Under the framework of structural time series models I noticed that the default critical value based on the sample size is too high, since all the potential outliers located in the first stage were discarded in the second stage (even in simulated series with known location of outliers). In order to investigate this issue, the argument discard_cval has been added. In this way a different critical value can be used in the second stage. Alternatively, the argument discard_cval could be omitted and simply choose a lower critical value, cval, to be used in both stages. However, using the argument discard_cval is more convenient since it avoids locating too many outliers in the first stage. discard_cval is not affected by cval_reduce.
        :param tsmethod: The framework for time series modelling. It basically is the name of the function to which the arguments defined in args_tsmethod are referred to.
        :param fit_args: Additional arguments besides endog and exog to be passed into fit() method.
        :param args_tsmethod: An optional dictionary containing arguments to be passed to the function invoking the method selected in tsmethod.
        :param log_file: It is the path to the file where tracking information is printed. Ignored if None.
        :param
        """
        self.types = types.copy()
        if 'delta' not in self.types.columns:
            self.types['delta'] = np.nan
        if 'min_n' not in self.types.columns:
            self.types['min_n'] = np.nan
        self.types.loc[(self.types['type'] == 'TC') & self.types['delta'].isna(),'delta'] = 0.7
        self.types.loc[(self.types['type'] == 'STC') & self.types['delta'].isna(),'delta'] = 1
        self.types.loc[(self.types['type'] == 'VC') & self.types['min_n'].isna(),'min_n'] = 20
        self.types = self.types.drop_duplicates()
        self.types['type_id'] = np.arange(len(self.types.index))

        self.maxit = maxit
        self.maxit_iloop = maxit_iloop
        self.maxit_oloop = maxit_oloop
        self.cval = cval
        self.cval_reduce = cval_reduce
        self.discard_method = discard_method
        self.discard_cval = discard_cval
        self.tsmethod = tsmethod

        if args_tsmethod is None:
            args_tsmethod = {}
            if tsmethod == 'AutoARIMA':
                args_tsmethod['information_criterion'] = 'bic'
            else:
                args_tsmethod['order'] = (1, 0, 1)
        if 'sarimax_kwargs' in args_tsmethod and 'trend_offset' in args_tsmethod['sarimax_kwargs']:
            self.trend_offset = args_tsmethod['sarimax_kwargs']['trend_offset']
        else:
            self.trend_offset = 1

        self.args_tsmethod = args_tsmethod

        ## Propogated later through other functions
        if self.tsmethod == 'AutoARIMA':
            self.ts_model = AutoARIMA(**self.args_tsmethod)
        else:
            self.ts_model = ARIMA(**self.args_tsmethod)
        self.fitted_trend = None
        self.poly_coeffs = None

    def get_resid(self) -> np.ndarray:
        """
        Get residuals of fitted model.

        :return: A 1D array of residual vector.
        """
        if self.tsmethod == 'AutoARIMA':
            return self.ts_model.model_.resid()
        else:
            return self.ts_model.resid()

    def get_order(self) -> Tuple[NonnegativeInt, NonnegativeInt, NonnegativeInt]:
        """
        Get order of fitted model (p, d, q).
        """
        if self.tsmethod == 'AutoARIMA':
            return self.ts_model.model_.order
        else:
            return self.ts_model.order

    def get_seasonal_order(self) -> Tuple[NonnegativeInt, NonnegativeInt, NonnegativeInt, NonnegativeInt]:
        """
        Get seasonal order of fitted model (P, D, Q, s).
        """
        if self.tsmethod == 'AutoARIMA':
            return self.ts_model.model_.seasonal_order
        else:
            return self.ts_model.seasonal_order

    def get_params(self) -> pd.DataFrame:
        """
        Get parameter of fitted ts_model.

        :return: Reformatted table of fitted parameters and the standard deviation.
        """
        if self.tsmethod == 'AutoARIMA':
            result = pd.DataFrame(self.ts_model.model_.arima_res_.summary().tables[1].data[1:], columns=self.ts_model.model_.arima_res_.summary().tables[1].data[0])
        else:
            result = pd.DataFrame(self.ts_model.arima_res_.summary().tables[1].data[1:], columns=self.ts_model.arima_res_.summary().tables[1].data[0])
        result.index = result['']
        result = result.drop(columns=['']).astype(float)
        return result

    def fit_ts_model(
        self,
        endog: pd.DataFrame,
        exog: Optional[pd.DataFrame],
        fit_args: Dict[str, Any],
        fix_order: bool=False):
        """
        Fit ts model according to tsmethod and fit_args.

        :param endog: a time series where outliers are to be detected.
        :param exog: an optional matrix of regressors with the same number of rows as y.
        :param fit_args: Additional arguments besides endog and exog to be passed into fit() method.
        :param fix_order: Only used when tsmethod is AutoARIMA, when true, the tuned order is preserved, only the parameters are updated.
        """
        if fit_args is None:
            fit_args = {}
        else:
            fit_args = fit_args.copy()
        fit_args['y'] = endog
        fit_args['X'] = exog
        if self.tsmethod == 'AutoARIMA' and fix_order:
            self.tsmethod.model_.fit(**fit_args)
        else:
            self.ts_model.fit(**fit_args) ## self.ts_model gets updated

    def outliers_tstats(self, sigma) -> pd.DataFrame:
        """
        Compute t-statistics for the significance of outliers.

        :param sigma: Standard deviation of residuals.
        :return: A dataframe containing the residuals and their estimated coefficient factor for each type and t-statistics.
        """
        order = self.get_order()
        seasonal_order = self.get_seasonal_order()
        params = self.get_params()['coef']
        resid = self.get_resid()
        types = self.types
        trend_offset = self.trend_offset

        seasonal_s = max(seasonal_order[3], 1)
        ar_params, diff_params, ma_params, ar_seasonal_params, diff_seasonal_params, ma_seasonal_params = sarima_params_to_poly_coeffs(params, order, seasonal_order)
        pi_coefs = params_to_infinite_representations(ar_params, diff_params, ma_params, ar_seasonal_params, diff_seasonal_params, ma_seasonal_params, leads=len(resid), right_on_left=False)
        ao_xy = signal.convolve(np.concatenate((resid, np.zeros(len(resid)-1))), np.flip(pi_coefs), method='direct')[(len(resid)-1):(1-len(resid))]
        rev_ao_xy = np.flip(ao_xy)

        result = pd.DataFrame({'residuals': resid, 't_index': np.arange(start=trend_offset, stop=trend_offset+len(resid))})
        result = pd.merge(result, types, how='cross')
        result['coefhat'] = np.nan
        result['tstat'] = np.nan
        for idx in types.index:
            if types.loc[idx,'type'] == 'AO':
                xxinv = np.flip(1 / np.cumsum(np.power(pi_coefs, 2)))
                coef_hat = ao_xy * xxinv
                result.loc[result['type'] == 'AO','coefhat'] = coef_hat
                result.loc[result['type'] == 'AO','tstat'] = coef_hat / (sigma * np.sqrt(xxinv))
            elif types.loc[idx,'type'] == 'TC':
                delta = types.loc[idx,'delta']
                x_y = np.flip(recursive_filter(rev_ao_xy, np.array([delta])))
                dinvf = recursive_filter(pi_coefs, np.array([delta]))
                xxinv = np.flip(1 / np.cumsum(np.power(dinvf, 2)))
                coef_hat = x_y * xxinv
                result.loc[result['type'] == 'TC','coefhat'] = coef_hat
                result.loc[result['type'] == 'TC','tstat'] = coef_hat / (sigma * np.sqrt(xxinv))
            elif types.loc[idx,'type'] == 'STC':
                delta = types.loc[idx,'delta']
                rm_id = np.arange(seasonal_s)
                x_y = np.flip(np.delete(diffinv(rev_ao_xy, lag=seasonal_s), rm_id))
                dinvf = np.delete(diffinv(pi_coefs, lag=seasonal_s), rm_id)
                xxinv = np.flip(1 / np.cumsum(np.power(dinvf, 2)))
                coef_hat = x_y * xxinv
                result.loc[result['type'] == 'STC','coefhat'] = coef_hat
                result.loc[result['type'] == 'STC','tstat'] = coef_hat / (sigma * np.sqrt(xxinv))
            elif types.loc[idx,'type'] == 'VC':
                min_n = types.loc[idx,'min_n']
                bt_sqrd = [(np.sum(np.power(resid[:(i-1)], 2)), np.sum(np.power(resid[i:], 2))) for i in range(min_n+1, len(resid)-min_n)]
                r_d = np.array([(bt_sqrd[i][1] * (i-1)) / (bt_sqrd[i][1] * (len(resid)-i+1)) for i in range(len(bt_sqrd))])
                coef_hat = np.zeros(len(resid))
                coef_hat[(min_n+1):(len(resid)-min_n)] = np.sqrt(r_d) - 1
                result.loc[result['type'] == 'VC','coefhat'] = coef_hat
                result.loc[result['type'] == 'VC','tstat'] = r_d.max() / r_d.min()
            else:
                result.loc[result['type'] == 'IO','coefhat'] = resid
                result.loc[result['type'] == 'IO','tstat'] = resid / sigma
        return result

    def locate_outliers(self, cval: PositiveFlt, id_start: NonnegativeInt):
        """
        Stage I of the Procedure: Locate Outliers (Baseline Function)

        Five general types of outliers can be considered. By default: "AO" additive outliers, and "TC" temporary changes are selected; "IO" innovative outliers; "STC" seasonal temporary shifts; "VC" variance change can also be selected. LS and SLS are special cases of TC and STC withd delta set to 1.

        :param cval: The critical value to determine the significance of each type of outlier.
        :param id_start: The starting index to be assigned to outliers once identified.
        :return: Identified outliers in dataframe format.
        """
        resid = self.get_resid()
        sigma = 1.483 * np.quantile(np.abs(resid - np.quantile(resid, 0.5)), 0.5) ## MAD estimation
        tmp = self.outliers_tstats(sigma) ## Quantile estimation of standard deviation
        identified_ol = tmp[tmp['tstat'].abs() > cval]
        identified_ol = identified_ol.groupby('t_index').apply(lambda x: x.iloc[x['tstat'].abs().argmax()]).reset_index(drop=True)
        identified_ol['id'] = range(id_start, id_start + len(identified_ol.index))
        return identified_ol

    def remove_consecutive_outliers(self, located_ol: pd.DataFrame):
        """
        Identify and remove consecutive outliers.

        :param located_ol: Output from function locate_outliers.
        :return: The outlier dataframe with consecutive outliers removed.
        """
        located_ol = located_ol.sort_values('t_index')
        located_ol['cscid'] = 1
        located_ol.loc[located_ol['t_index'].diff() == 1,'cscid'] = 0
        located_ol['cscid'] = located_ol['cscid'].cumsum()
        return located_ol.groupby('cscid').apply(lambda x: x.iloc[x['tstat'].abs().argmax()]).reset_index(drop=True).drop(columns=['cscid'])

    def outlier_effect_on_residuals(self, located_ol: pd.DataFrame) -> pd.DataFrame:
        """
        Get outliers effect on residuals. The VC outliers are ignored.

        :param located_ol: The output from remove_consecutive_outliers.
        :return: A 2d array indicating the outlier effect on residuals (rows), for each outlier (cols).
        """
        order = self.get_order()
        seasonal_order = self.get_seasonal_order()
        params = self.get_params()['coef']
        resid = self.get_resid()

        seasonal_s = max(seasonal_order[3], 1)
        ar_params, diff_params, ma_params, ar_seasonal_params, diff_seasonal_params, ma_seasonal_params = sarima_params_to_poly_coeffs(params, order, seasonal_order)
        pi_coefs = params_to_infinite_representations(ar_params, diff_params, ma_params, ar_seasonal_params, diff_seasonal_params, ma_seasonal_params, leads=len(resid), right_on_left=False)
        tc_ma_coefs = {}
        for delta in located_ol[located_ol['type'] == 'TC']['delta'].unique():
            delta_term = np.array([-delta, 1])
            tc_ma_coefs[delta] = params_to_infinite_representations(ar_params, diff_params, np.polymul(ma_params, delta_term), ar_seasonal_params, diff_seasonal_params, ma_seasonal_params, leads=len(resid), right_on_left=False)
        seasonal_tc_ma_coefs = {}
        for delta in located_ol[located_ol['type'] == 'STC']['delta'].unique():
            delta_term = np.concatenate((np.array([-delta]), np.zeros(seasonal_s-1), np.array([1])))
            seasonal_tc_ma_coefs[delta] = params_to_infinite_representations(ar_params, diff_params, ma_params, ar_seasonal_params, diff_seasonal_params, np.polymul(ma_seasonal_params, delta_term), leads=len(resid), right_on_left=False)

        def xreg_io(indices, weights):
            matrix_i = np.zeros((len(resid), len(indices)))
            matrix_i[indices,:] = np.diag(weights)
            return matrix_i

        def xreg_ao(indices, weights):
            matrix_i = np.zeros((len(resid), len(indices)))
            matrix_i[indices,:] = np.eye(len(indices))
            for i in range(len(indices)):
                matrix_i[:,i] = weights.iloc[i] * signal.convolve(np.concatenate((np.zeros(len(resid)-1), matrix_i[:,i])), pi_coefs, method='direct')[(matrix_i.shape[0]-1):(1-matrix_i.shape[0])]
            return matrix_i

        def xreg_tc(indices, weights, delta):
            matrix_i = np.zeros((len(resid), len(indices)))
            matrix_i[indices,:] = np.eye(len(indices))
            updated_pi_coefs = tc_ma_coefs[delta]
            for i in range(len(indices)):
                matrix_i[:,i] = weights.iloc[i] * signal.convolve(np.concatenate((np.zeros(len(resid)-1), matrix_i[:,i])), updated_pi_coefs, method='direct')[(matrix_i.shape[0]-1):(1-matrix_i.shape[0])]
            return matrix_i

        def xreg_stc(indices, weights, delta):
            matrix_i = np.zeros((len(resid), len(indices)))
            matrix_i[indices,:] = np.eye(len(indices))
            updated_pi_coefs = seasonal_tc_ma_coefs[delta]
            for i in range(len(indices)):
                matrix_i[:,i] = weights.iloc[i] * signal.convolve(np.concatenate((np.zeros(len(resid)-1), matrix_i[:,i])), updated_pi_coefs, method='direct')[(matrix_i.shape[0]-1):(1-matrix_i.shape[0])]
            return matrix_i

        result = []
        for ol_type, temp_df in located_ol.groupby(['type', 'delta', 'min_n', 'type_id'], dropna=False):
            if ol_type[0] == 'AO':
                result.append(xreg_ao(indices=temp_df['t_index'].to_numpy()-1, weights=temp_df['coefhat'].to_numpy()))
            elif ol_type[0] == 'IO':
                result.append(xreg_io(indices=temp_df['t_index'].to_numpy()-1, weights=temp_df['coefhat'].to_numpy()))
            elif ol_type[0] == 'TC':
                result.append(xreg_tc(indices=temp_df['t_index'].to_numpy()-1, weights=temp_df['coefhat'].to_numpy(), delta=ol_type[1]))
            elif ol_type[0] == 'STC':
                result.append(xreg_stc(indices=temp_df['t_index'].to_numpy()-1, weights=temp_df['coefhat'].to_numpy(), delta=ol_type[1]))
        result = np.concatenate(result, axis=1)
        result = pd.DataFrame(result, columns='ol_id_' + located_ol['id'].astype(str), index=range(result.shape[0]))
        return result

    def outlier_effect_on_responses(self, endog, located_ol: pd.DataFrame, use_fitted_coefs: bool=True) -> pd.DataFrame:
        """
        Get outliers effect on responses. The VC outliers are ignored.

        This function can be used to find outlier effects on responses, or construct outlier effects as exogenous regressor.

        :param endog: Response time series.
        :param located_ol: The output from remove_consecutive_outliers.
        :param use_fitted_coefs: Whether to use fitted coefficients as weights. If false, then weights are set to 1.
        :return: A 2d array indicating the outlier effect on residuals (rows), for each outlier (cols).
        """
        order = self.get_order()
        seasonal_order = self.get_seasonal_order()
        params = self.get_params()['coef']

        seasonal_s = max(seasonal_order[3], 1)
        ar_params, diff_params, ma_params, ar_seasonal_params, diff_seasonal_params, ma_seasonal_params = sarima_params_to_poly_coeffs(params, order, seasonal_order)
        psi_coefs = params_to_infinite_representations(ar_params, diff_params, ma_params, ar_seasonal_params, diff_seasonal_params, ma_seasonal_params, leads=len(endog), right_on_left=True)

        def xreg_io(indices, weights):
            matrix_i = np.zeros((len(endog), len(indices)))
            matrix_i[indices,:] = np.eye(len(indices))
            for i in range(len(indices)):
                matrix_i[:,i] = weights[i] * signal.convolve(np.concatenate((np.zeros(len(endog)-1), matrix_i[:,i])), psi_coefs, method='direct')[(matrix_i.shape[0]-1):(1-matrix_i.shape[0])]
            return matrix_i

        def xreg_ao(indices, weights):
            matrix_i = np.zeros((len(endog), len(indices)))
            matrix_i[indices,:] = np.diag(weights)
            return matrix_i

        def xreg_tc(indices, weights, delta):
            matrix_i = np.zeros((len(endog), len(indices)))
            matrix_i[indices,:] = np.eye(len(indices))
            for i in range(len(indices)):
                matrix_i[:,i] = weights[i] * recursive_filter(matrix_i[:,i], np.array([delta]))
            return matrix_i

        def xreg_stc(indices, weights, delta):
            matrix_i = np.zeros((len(endog), len(indices)))
            matrix_i[indices,:] = np.diag(weights)
            for i in range(len(indices)):
                matrix_i[:,i] = recursive_filter(matrix_i[:,i], np.concatenate((np.zeros(seasonal_s-1), np.array([delta]))))
            return matrix_i

        result = []
        for ol_type, temp_df in located_ol.groupby(['type', 'delta', 'min_n', 'type_id'], dropna=False):
            if ol_type[0] == 'AO':
                if not use_fitted_coefs:
                    result.append(xreg_ao(indices=temp_df['t_index'].to_numpy()-1, weights=np.ones(len(temp_df.index))))
                else:
                    result.append(xreg_ao(indices=temp_df['t_index'].to_numpy()-1, weights=temp_df['coefhat'].to_numpy()))
            elif ol_type[0] == 'IO':
                if not use_fitted_coefs:
                    result.append(xreg_io(indices=temp_df['t_index'].to_numpy()-1, weights=np.ones(len(temp_df.index))))
                else:
                    result.append(xreg_io(indices=temp_df['t_index'].to_numpy()-1, weights=temp_df['coefhat'].to_numpy()))
            elif ol_type[0] == 'TC':
                if not use_fitted_coefs:
                    result.append(xreg_tc(indices=temp_df['t_index'].to_numpy()-1, weights=np.ones(len(temp_df.index)), delta=ol_type[1]))
                else:
                    result.append(xreg_tc(indices=temp_df['t_index'].to_numpy()-1, weights=temp_df['coefhat'].to_numpy(), delta=ol_type[1]))
            elif ol_type[0] == 'STC':
                if not use_fitted_coefs:
                    result.append(xreg_stc(indices=temp_df['t_index'].to_numpy()-1, weights=np.ones(len(temp_df.index)), delta=ol_type[1]))
                else:
                    result.append(xreg_stc(indices=temp_df['t_index'].to_numpy()-1, weights=temp_df['coefhat'].to_numpy(), delta=ol_type[1]))
        result = np.concatenate(result, axis=1)
        result = pd.DataFrame(result, columns='ol_id_' + located_ol['id'].astype(str), index=range(result.shape[0]))
        return result

    def locate_outlier_iloop(self, cval: PositiveFlt, id_start: NonnegativeInt):
        """
        Locate outliers inner loop helper.

        :param cval: The critical value to determine the significance of each type of outlier.
        :param id_start: The starting index to be assigned to outliers once identified.
        :return: Identified outliers in dataframe format.
        """
        resid = self.get_resid()

        result = pd.DataFrame(columns=['type_id', 'id', 'type', 'residuals', 't_index', 'coefhat', 'tstat', 'delta', 'min_n'])
        its = 0
        while its < self.maxit_iloop:
            located_ol = self.locate_outliers(cval, id_start)

            located_ol = located_ol[located_ol['type'] != 'VC'] ## TODO: Add VC support remove VC from residuals

            located_ol = located_ol.groupby('type_id').apply(self.remove_consecutive_outliers).reset_index(drop=True)

            located_ol = located_ol[~located_ol['t_index'].isin(result['t_index'])]

            if len(located_ol.index) == 0:
                break

            result = pd.concat([result, located_ol], axis=0)

            ol_effect_matrix = self.outlier_effect_on_residuals(located_ol)

            resid -= ol_effect_matrix.sum(axis=1)

            its += 1

        if its == self.maxit_iloop:
            warnings.warn('Maximum number of iterations reached for inner loop.')
        return result

    def zero_inflated_first_resid(self):
        """
        Helper function for zeroing the first d + seaonal_q * seasonal_d if they are overinflated.
        """
        order = self.get_order()
        seasonal_order = self.get_seasonal_order()
        resid = self.get_resid()

        tmp = order[1] + seasonal_order[3] * seasonal_order[1]
        if tmp > 1:
            id0resid = list(range(0, tmp))
        else:
            id0resid = [0, 1]

        if (np.abs(resid[id0resid]) > 3.5 * np.std(np.delete(resid, id0resid))).any():
            resid[id0resid] = 0

    def locate_outlier_oloop(
        self,
        endog: pd.DataFrame,
        exog: Optional[pd.DataFrame],
        cval: PositiveFlt,
        id_start: NonnegativeInt,
        fit_args: Optional[Dict[str, Any]]=None) -> pd.DataFrame:
        """
        Locate outliers outer loop helper.

        :param endog: a time series where outliers are to be detected.
        :param exog: an optional matrix of regressors with the same number of rows as y.
        :param cval: The critical value to determine the significance of each type of outlier.
        :param id_start: The starting index to be assigned to outliers once identified.
        :param fit_args: The arguments passed into fit() method.
        :return: Located outliers in a dataframe.
        """
        endog = endog.copy()

        result = pd.DataFrame(columns=['type_id', 'id', 'type', 'residuals', 't_index', 'coefhat', 'tstat', 'delta', 'min_n'])

        its = 0
        while its < self.maxit_oloop:
            self.zero_inflated_first_resid()

            inner_result = self.locate_outlier_iloop(cval, id_start)

            inner_result = inner_result[inner_result['type'] != 'VC'] ## TODO: Add VC support remove VC from response

            inner_result = self.remove_consecutive_outliers(inner_result)

            inner_result = inner_result[~inner_result['t_index'].isin(result['t_index'])]

            if len(inner_result.index) == 0:
                break

            result = pd.concat([result, inner_result], axis=0)

            ol_effect_matrix = self.outlier_effect_on_responses(endog, inner_result, True)

            endog -= ol_effect_matrix.sum(axis=1)

            self.fit_ts_model(endog, exog, fit_args, True)

            its += 1

        if its == self.maxit_oloop:
            warnings.warn('Maximum number of iterations reached for outer loop.')
        return result

    def discard_outliers(
        self,
        located_ol: pd.DataFrame,
        endog: pd.DataFrame,
        exog: Optional[pd.DataFrame],
        cval: PositiveFlt,
        fit_args: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        This functions tests for the significance of a given set of outliers in a time series model that is fitted including the outliers as regressor variables.

        :param located_ol: Output from locate_outlier_oloop.
        :param endog: a time series where outliers are to be detected.
        :param exog: an optional matrix of regressors with the same number of rows as y.
        :param cval: The critical value to determine the significance of each type of outlier.
        :param fit_args: The arguments passed into fit() method.
        :return: A tuple of three dataframe: fitted outliers, adjusted endogenous series, and adjusted exogenous series.
        """
        located_ol = self.remove_consecutive_outliers(located_ol)
        xreg = self.outlier_effect_on_responses(endog, located_ol, False)

        if exog is not None:
            xreg = pd.concat([exog, xreg], axis=1)

        its = 0
        if self.discard_method == 'en-masse':
            while True:
                self.fit_ts_model(endog, xreg, fit_args, True)
                param_table = self.get_params()
                param_table['tstat'] = param_table['coef'] / param_table['std err']

                rm_ol_table = param_table[param_table[''].str.match(r'ol_id_\d+') & (param_table['tstat'].abs() < cval)]

                if len(rm_ol_table.index) > 0:
                    located_ol = located_ol[~located_ol['type'].isin(rm_ol_table[''])]
                    located_ol = pd.merge(located_ol.drop(columns=['tstat', 'coefhat']), param_table[['', 'tstat', 'coef']].rename(columns={'': 'type', 'coef': 'coefhat'}))
                    xreg = xreg.drop(columns=rm_ol_table[''])
                else:
                    break

                its += 1
        else:
            located_ol = located_ol.sort_values(['tstat'], ascending=False, key=abs).reset_index(drop=True)

            xregaux = pd.DataFrame(index=xreg.index)
            for i in located_ol.index:
                xregaux = pd.concat([xregaux, xreg.loc[:,located_ol.loc[i,'type']]], axis=1)

                self.fit_ts_model(endog, xregaux, fit_args, True)
                param_table = self.get_params()

                param_table['tstat'] = param_table['coef'] / param_table['std err']

                rm_ol_table = param_table[param_table[''].str.match(r'ol_id_\d+') & (param_table['tstat'].abs() < cval)]

                if len(rm_ol_table.index) > 0:
                    located_ol = located_ol[~located_ol['type'].isin(rm_ol_table[''])]
                    located_ol = pd.merge(located_ol.drop(columns=['tstat', 'coefhat']), param_table[['', 'tstat', 'coef']].rename(columns={'': 'type', 'coef': 'coefhat'}))
                    xregaux = xregaux.drop(columns=rm_ol_table[''])
            xreg = xregaux

        ## Adjust endog: TODO: include VC
        ol_effect = self.outlier_effect_on_responses(endog, located_ol, True)
        adj_endog = endog - ol_effect.sum(axis=1)

        return located_ol, adj_endog, xreg

    def fit(self, endog, exog, fit_args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Automatic Procedure for Detection of Outliers.

        :param endog: a time series where outliers are to be detected.
        :param exog: an optional matrix of regressors with the same number of rows as y.
        :param fit_args: Additional arguments besides endog and exog to be passed into fit() method.
        """
        if self.cval is None:
            if len(endog.index) <= 50:
                self.cval = 3
            elif len(endog.index) >= 450:
                self.cval = 4
            else:
                self.cval = 3 + 0.0025 * (len(endog.index) - 50)

        if self.discard_cval is None:
            self.discard_cval = self.cval

        cval0 = self.cval

        its = 0
        result = pd.DataFrame(columns=['type_id', 'id', 'type', 'residuals', 't_index', 'coefhat', 'tstat', 'delta', 'min_n'])
        start_id = 0
        while its < self.maxit:
            self.fit_ts_model(endog, exog, fit_args, False)
            located_ol = self.locate_outlier_oloop(endog, exog, cval0, start_id, fit_args)
            if len(located_ol.index) > 0:
                located_ol, endog, exog = self.discard_outliers(located_ol, endog, exog, self.discard_cval, fit_args)
                result = pd.concat([result, located_ol], axis=0)
                start_id = result['id'].max() + 1
            else:
                break

            its += 1
            cval0 = cval0 * (1 - self.cval_reduce)

        return result, endog, exog