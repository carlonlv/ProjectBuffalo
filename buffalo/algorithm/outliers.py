"""
This module contains algorithms for identifying/removing/predicting outliers.
"""
from functools import reduce
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pmdarima import ARIMA, AutoARIMA
from scipy import signal

from ..utility import PositiveFlt, PositiveInt, expand_grid


def find_poly_time_trend(ts_model: ARIMA, trend_offset: PositiveFlt, nobs: PositiveFlt):
    """ 
    Find fitted polynomial time trend.
    
    :return: Fitted polynomial time trend which is equal in size of input endogenous time series.
    """
    params = ts_model.params()
    time_obs = np.arange(start=trend_offset, stop=trend_offset+nobs)

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

def sarima_params_to_poly_coeffs(ts_model: ARIMA) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """ 
    Convert coefficients to polynomial cofficients for AR, MA, sAR, sMA parameters.

    :return: Converted polynomials for AR polynomial, MA polynomial, sAR polynomial, sMA polynomial.
    """
    def fill_poly_with_zero(params):
        for i in range(params.index.max()):
            if i not in params:
                params[i] = 0

    params = ts_model.params()

    difference_d = ts_model.order[1]
    difference_seasonal_d = ts_model.seasonal_order[1]
    seasonal_s = max(ts_model.seasonal_order[3], 1)

    ar_params = params[params.index.str.match(r'ar\.L\d+')]
    ar_params *= -1
    ar_params.index = ar_params.index.str.replace(r'ar\.L', '').astype(int)
    ar_params[0] = 1
    ar_params = ar_params.sort_index(ascending=False) ## Decreasing order in polynomial
    fill_poly_with_zero(ar_params)

    ar_seasonal_params = params[params.index.str.match(r'ar\.S\.L\d+')]
    ar_seasonal_params *= -1
    ar_seasonal_params.index = ar_seasonal_params.index.str.replace(r'ar\.S\.L', '').astype(int)
    ar_seasonal_params[0] = 1
    ar_seasonal_params.index *= seasonal_s
    ar_seasonal_params = ar_seasonal_params.sort_index(ascending=False)
    fill_poly_with_zero(ar_seasonal_params)

    diff_params = reduce(np.polymul, [np.array([-1, 1])] * difference_d, np.array([1]))

    ma_params = params[params.index.str.match(r'ma\.L\d+')]
    ma_params.index = ma_params.index.str.replace(r'ma\.L', '').astype(int)
    ma_params[0] = 1
    ma_params = ma_params.sort_index(ascending=False) ## Decreasing order in polynomial
    fill_poly_with_zero(ma_params)

    ma_seasonal_params = params[params.index.str.match(r'ma\.S\.L\d+')]
    ma_seasonal_params.index = ma_seasonal_params.index.str.replace(r'ma\.S\.L', '').astype(int)
    ma_seasonal_params[0] = 1
    ma_seasonal_params.index *= seasonal_s
    ma_seasonal_params = ma_seasonal_params.sort_index(ascending=False)
    fill_poly_with_zero(ma_seasonal_params)

    diff_seasonal_params = reduce(np.polymul, [np.concatenate((np.array([-1]), np.zeros(seasonal_s-1), np.array([1])))] * difference_seasonal_d, np.array([1]))

    return ar_params, diff_params, ma_params, ar_seasonal_params, diff_seasonal_params, ma_seasonal_params

def find_intercept(ts_model: ARIMA, trend_offset: PositiveFlt, nobs: PositiveFlt) -> np.ndarray:
    """
    Convert fitted trend to infinite MA representation. Can remove this value from original time series, the residuals is fitted by SARIMAX model.
    """
    ar_params, _, diff_params, ar_seasonal_params, diff_seasonal_params, _ = sarima_params_to_poly_coeffs(ts_model)

    left_params = reduce(np.polymul, [ar_params, diff_params, ar_seasonal_params, diff_seasonal_params])

    time_trend = find_poly_time_trend(ts_model, trend_offset, nobs)
    return time_trend / left_params.sum()

def params_to_infinite_representations(ts_model: ARIMA, leads: PositiveInt=100, right_on_left: bool=True) -> np.ndarray:
    """
    Convert fitted parameters to infinite MA representation.

    :param ts_model:
    :param leads: Truncate this number of polynomials to represent infinite ma representations.
    :return: Infinite MA(right_on_left) or AR(left_on_right) series in increasing polynomials.
    """
    ar_params, diff_params, ma_params, ar_seasonal_params, diff_seasonal_params, ma_seasonal_params = sarima_params_to_poly_coeffs(ts_model)

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

def outliers_tstats(ts_model: ARIMA, types: List[Literal['AO', 'LS', 'TC']]=["AO", "LS", "TC"], sigma: Optional[PositiveFlt]=None, delta: PositiveInt=0.7):
    """
    Compute t-statistics for the significance of outliers.
    """
    res = ts_model.resid()
    pi_coefs = params_to_infinite_representations(ts_model, leads=len(res), right_on_left=False)
    ao_xy = signal.convolve(np.concatenate((res, np.zeros(len(res)-1))), np.flip(pi_coefs))[(len(res)-1):(1-len(res))]
    rev_ao_xy = np.flip(ao_xy)

    result = expand_grid(residuals = res, type = types)
    result['coefhat'] = np.nan
    result['tstat'] = np.nan
    for _ in types:
        if _ == 'AO':
            xxinv = np.flip(1 / np.cumsum(np.power(pi_coefs, 2)))
            coef_hat = ao_xy * xxinv
            result.loc[result['type'] == 'AO','coefhat'] = coef_hat
            result.loc[result['type'] == 'AO','tstat'] = coef_hat / (sigma * np.sqrt(xxinv))
        elif _ == 'TC':
            pass
        elif _ == 'LS':
            pass
        else:
            result.loc[result['type'] == 'IO','coefhat'] = res
            result.loc[result['type'] == 'AO','tstat'] = res / sigma
    return

def compute_tstats(params: pd.Series, resid: np.ndarray, types: List[Literal['AO', 'LS', 'TC']], sigma: PositiveFlt, delta: PositiveFlt):
    """
    This function applies the t-statistics for the significance of outliers at every time point and selects those that are significant given a critical value.

    
    """
    
    return

def locate_outliers(resid: np.ndarray, params: pd.Series, cval: PositiveFlt=3.5, types: List[Literal['AO', 'LS', 'TC']]=['AO', 'LS', 'TC'], delta: PositiveFlt=0.7):
    """
    Stage I of the Procedure: Locate Outliers (Baseline Function)

    Five types of outliers can be considered. By default: "AO" additive outliers, "LS" level shifts, and "TC" temporary changes are selected; "IO" innovative outliers; "SLS" seasonal level shifts; "VC" variance change can also be selected.

    :param resid: Residuals from a time series model fitted to the data.
    :param params: Containing the parameters of the model fitted to the data. See details below.
    :param cval: The critical value to determine the significance of each type of outlier.
    :param types: A character vector indicating the types of outliers to be considered.
    :param delta: Parameter of the temporary change type of outlier. 
    :return:
    """
    sigma = 1.483 * np.quantile(np.abs(resid - np.quantile(resid, 0.5)), 0.5)
    


  tmp <- outliers.tstatistics(pars = pars, resid = resid, 
    types = types, sigma = sigma, delta = delta)
  ind <- which(abs(tmp[, , "tstat", drop = FALSE]) > cval, 
    arr.ind = TRUE)
  mo <- data.frame(factor(gsub("^(.*)tstats$", "\\1", dimnames(tmp)[[2]][ind[, 
    2]]), levels = c("IO", "AO", "LS", "TC", "SLS")), ind[, 
    1], tmp[, , "coefhat", drop = FALSE][ind], tmp[, , "tstat", 
    drop = FALSE][ind])
  colnames(mo) <- c("type", "ind", "coefhat", "tstat")
  if (nrow(ind) == 1) 
    rownames(mo) <- NULL
  ref <- unique(mo[, "ind"][duplicated(mo[, "ind"])])
  for (i in ref) {
    ind <- which(mo[, "ind"] == i)
    moind <- mo[ind, ]
    mo <- mo[-ind[-which.max(abs(moind[, "tstat"]))], ]
  }
  mo
    return

def locate_outlier_iloop(ts_model:ARIMA, cval: int = 3.5, types: List[Literal['AO', 'LS', 'TC']]=['AO', 'LS', 'TC'], maxit: int=4, delta: int=0.7):
    """
    Locate outliers inner loop helper.

    :param resid:
    :param params:
    :param cval:
    :param types:
    :param maxit:
    :param delta:
    """
    resid = ts_model.resid()
    its = 0
    while its < maxit:
        mo = locat
        its += 1
    return

def locate_outlier_oloop():
    """
    """
    return

class IterativeTtestOutlierDetection:
    """
    This class implements the traditional iterative procedure for identifying and removeing outliers.
    Five types of outliers can be considered. By default: "AO" additive outliers, "LS" level shifts, and "TC" temporary changes are selected; "IO" innovative outliers and "SLS" seasonal level shifts can also be selected.
    The algroithm iterates around locating and removing outliers first for the original series and then for the adjusted series. The process stops if no additional outliers are found in the current iteration or if maxit iterations are reached.
    """

    def __init__(
        self,
        endog,
        exog: Optional[pd.DataFrame]=None,
        cval: Optional[PositiveFlt]=None,
        delta: PositiveFlt = 0.7,
        types: Optional[List[Literal['IO', 'AO', 'LS', 'TC', 'SLS', 'VC']]]=None,
        maxit: PositiveInt=1,
        maxit_iloop: PositiveInt=4,
        maxit_oloop: PositiveInt=4,
        cval_reduce: PositiveFlt=0.14286,
        discard_method: Literal['en-masse', 'bottom-up']='en_masse',
        discard_cval: Optional[PositiveFlt]=None,
        tsmethod: Literal["AutoARIMA", "ARIMA"]='auto.arima',
        args_tsmethod: Optional[Dict[str, Any]]=None,
        check_rank: bool=True) -> None:
        """
        Initializer and configuration for IterativeTtestOutlierDetection.

        :param endog: a time series where outliers are to be detected.
        :param exog: an optional matrix of regressors with the same number of rows as y.
        :param cval: The critical value to determine the significance of each type of outlier. If no value is specified for argument cval a default value based on the sample size is used. Let nn be the number of observations. If $n \leq 50$, then cval is set equal to 3.0; If $n \geq 450$, then cval is set equal to 4.0; otherwise cval is set equal to $3 + 0.0025 * (n - 50)3 + 0.0025 * (n - 50)$.
        :param delta: Parameter of the temporary change type of outlier.
        :param types: A character list indicating the type of outlier to be considered by the detection procedure: innovational outliers ("IO"), additive outliers ("AO"), level shifts ("LS"), temporary changes ("TC") and seasonal level shifts ("SLS"). If None is provided, then a list of 'AO', 'LS', 'TC' is used.
        :param maxit: The maximum number of iterations.
        :param maxit_iloop: The maximum number of iterations in the inner loop. See locate_outliers.
        :param maxit_oloop: The maximum number of iterations in the outer loop.
        :param cval_reduce: Factor by which cval is reduced if the procedure is run on the adjusted series, if $maxit > 1$, the new critical value is defined as $cval * (1 - cval.reduce)$.
        :param discard_method: The method used in the second stage of the procedure. See discard.outliers.
        :param discard_cval: The critical value to determine the significance of each type of outlier in the second stage of the procedure (discard outliers). By default, the same critical value is used in the first stage of the procedure (location of outliers) and in the second stage (discard outliers). Under the framework of structural time series models I noticed that the default critical value based on the sample size is too high, since all the potential outliers located in the first stage were discarded in the second stage (even in simulated series with known location of outliers). In order to investigate this issue, the argument discard_cval has been added. In this way a different critical value can be used in the second stage. Alternatively, the argument discard_cval could be omitted and simply choose a lower critical value, cval, to be used in both stages. However, using the argument discard_cval is more convenient since it avoids locating too many outliers in the first stage. discard_cval is not affected by cval_reduce.
        :param tsmethod: The framework for time series modelling. It basically is the name of the function to which the arguments defined in args_tsmethod are referred to.
        :param args_tsmethod: An optional dictionary containing arguments to be passed to the function invoking the method selected in tsmethod.
        :param log_file: It is the path to the file where tracking information is printed. Ignored if None.
        :param
        """
        self.endog = endog
        self.exog = exog
        if cval is None:
            if len(self.endog.index) <= 50:
                cval = 3
            elif len(self.endog.index) >= 450:
                cval = 4
            else:
                cval = 3 + 0.0025 * (len(self.endog.index) - 50)
        self.cval = cval
        self.delta = delta
        if types is None:
            types = ['AO', 'LS', 'TC']
        self.types = types
        self.maxit = maxit
        self.maxit_iloop = maxit_iloop
        self.maxit_oloop = maxit_oloop
        self.cval_reduce = cval_reduce
        self.discard_method = discard_method
        if discard_cval is None:
            discard_cval = self.cval
        self.discard_cval = discard_cval
        self.tsmethod = tsmethod
        if args_tsmethod is None:
            args_tsmethod = {}
            if tsmethod == 'AutoARIMA':
                if exog is not None:
                    args_tsmethod['information_criterion'] = 'bic'
            else:
                if exog is not None:
                    args_tsmethod['order'] = (0, 1, 1)

        self.args_tsmethod = args_tsmethod
        self.check_rank = check_rank

        ## Propogated later through other functions
        self.ts_model = None
        self.fitted_trend = None
        self.poly_coeffs = None

    def _find_poly_time_trend(self) -> np.ndarray:
        """ 
        Find fitted polynomial time trend.

        :return: Fitted polynomial time trend which is equal in size of input endogenous time series.
        """
        params = self.ts_model.params()
        trend_offset = 1
        if 'trend_offset' in self.args_tsmethod:
            trend_offset = self.args_tsmethod['trend_offset']
        nobs = self.ts_model.arima_res_.nobs

        time_obs = np.arange(start=trend_offset, stop=trend_offset+nobs)

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

        self.fitted_trend = fitted_trend
        return fitted_trend

    def _sarima_params_to_poly_coeffs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ 
        Convert coefficients to polynomial cofficients for AR, MA, sAR, sMA parameters.

        :return: Converted polynomials for AR polynomial, MA polynomial, sAR polynomial, sMA polynomial.
        """
        def fill_poly_with_zero(params):
            for i in range(params.index.max()):
                if i not in params:
                    params[i] = 0

        params = self.ts_model.params()

        difference_d = self.ts_model.order[1]
        difference_seasonal_d = self.ts_model.seasonal_order[1]
        seasonal_s = max(self.ts_model.seasonal_order[3], 1)

        ar_params = params[params.index.str.match(r'ar\.L\d+')]
        ar_params *= -1
        ar_params.index = ar_params.index.str.replace(r'ar\.L', '').astype(int)
        ar_params[0] = 1
        ar_params = ar_params.sort_index(ascending=False) ## Decreasing order in polynomial
        fill_poly_with_zero(ar_params)

        ar_seasonal_params = params[params.index.str.match(r'ar\.S\.L\d+')]
        ar_seasonal_params *= -1
        ar_seasonal_params.index = ar_seasonal_params.index.str.replace(r'ar\.S\.L', '').astype(int)
        ar_seasonal_params[0] = 1
        ar_seasonal_params.index *= seasonal_s
        ar_seasonal_params = ar_seasonal_params.sort_index(ascending=False)
        fill_poly_with_zero(ar_seasonal_params)

        diff_params = reduce(np.polymul, [np.array([-1, 1])] * difference_d, np.array([1]))

        ma_params = params[params.index.str.match(r'ma\.L\d+')]
        ma_params.index = ma_params.index.str.replace(r'ma\.L', '').astype(int)
        ma_params[0] = 1
        ma_params = ma_params.sort_index(ascending=False) ## Decreasing order in polynomial
        fill_poly_with_zero(ma_params)

        ma_seasonal_params = params[params.index.str.match(r'ma\.S\.L\d+')]
        ma_seasonal_params.index = ma_seasonal_params.index.str.replace(r'ma\.S\.L', '').astype(int)
        ma_seasonal_params[0] = 1
        ma_seasonal_params.index *= seasonal_s
        ma_seasonal_params = ma_seasonal_params.sort_index(ascending=False)
        fill_poly_with_zero(ma_seasonal_params)

        diff_seasonal_params = reduce(np.polymul, [np.concatenate((np.array([-1]), np.zeros(seasonal_s-1), np.array([1])))] * difference_seasonal_d, np.array([1]))

        self.poly_coeffs = {
            'ar_params': ar_params,
            'diff_params': diff_params,
            'ma_params': ma_params,
            'ar_seasonal_params': ar_seasonal_params,
            'diff_seasonal_params': diff_seasonal_params,
            'ma_seasonal_params': ma_seasonal_params
        }
        return ar_params, diff_params, ma_params, ar_seasonal_params, diff_seasonal_params, ma_seasonal_params

    def _trend_to_intercept(self) -> np.ndarray:
        """
        Convert fitted trend to infinite MA representation.
        """
        ar_params, _, diff_params, ar_seasonal_params, diff_seasonal_params, _ = self._sarima_params_to_poly_coeffs()

        left_params = reduce(np.polymul, [ar_params, diff_params, ar_seasonal_params, diff_seasonal_params])

        time_trend = self._find_poly_time_trend()
        return time_trend / left_params.sum()

    def _params_to_ma(self, leads=100):
        """
        Convert fitted parameters to infinite MA representation.

        :param leads: Truncate this number of polynomials to represent infinite ma representations.
        """
        ar_params, diff_params, ma_params, ar_seasonal_params, diff_seasonal_params, ma_seasonal_params = self._sarima_params_to_poly_coeffs()

        left_params = reduce(np.polymul, [ar_params, diff_params, ar_seasonal_params, diff_seasonal_params])

        right_params = reduce(np.polymul, [ma_params, ma_seasonal_params])

        impluse = np.zeros(leads)
        impluse[0] = 1
        return signal.lfilter(right_params, left_params, impluse)

    def _locate_outlier_iloop(self, ):
        return pd.DataFrame()

    def _locate_outlier_oloop(self):
        """
        y, fit, types = c("AO", "LS", "TC"), cval = NULL, 
  maxit.iloop = 4, maxit.oloop = 4, delta = 0.7, logfile = NULL

        :param y:
        :param fit:
        :param types:
        :param cval
        """
        moall = pd.DataFrame(columns=['type', 'ind', 'coefhat', 'tstat'])
        tmp = self.ts_model.order[2] + self.ts_model.seasonal_order[3] * self.ts_model.seasonal_order[1]
        if tmp > 1:
            id0resid = list(range(0, tmp))
        else:
            id0resid = [0, 1]

        its = 0
        while its < self.maxit_oloop:
            res = self.ts_model.resid()
            if (res[id0resid] > 3.5 * np.std(np.delete(res, id0resid))).any():
                res[id0resid] = 0
            mo = self._locate_outlier_iloop()
            if len(mo.index) == 0:
                break
            its += 1

    def _fit(self):
        """
        """
        if self.tsmethod == 'AutoARIMA':
            self.ts_model = AutoARIMA(**self.args_tsmethod).fit(y=self.endog, X=self.exog).model_
        else:
            self.ts_model = ARIMA(**self.args_tsmethod).fit(y=self.endog, X=self.exog)

    def fit(self):
        """
        """
        cval0 = self.cval
        self._fit()

        return
