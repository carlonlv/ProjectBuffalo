"""
This module contains algorithms for identifying/removing/predicting outliers.
"""

from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import numpy as np
from pmdarima import AutoARIMA, ARIMA

from ..utility import PositiveFlt, PositiveInt


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
        types: Optional[List[Literal['IO', 'AO', 'LS', 'TC', 'SLS']]]=None,
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

    @staticmethod
    def _coef2poly(model: ARIMA, add: bool=True):
        """ Convert coefficients of ARIMA model to polynomials. This function collapses the polynomials of an ARIMA model into two polynomials: the product of the autoregressive polynomials and the product of the moving average polynomials.

        :param model: an object of class ARIMA.
        :param add: If TRUE, the polynomial of the differencing filter (if present in the model) is multiplied by the stationary autoregressive polynomial. Otherwise only the coefficients of the product of the stationary polynomials is returned.
        :return: A dictionary containing the elements: arcoefs, the coefficients of the product of the autoregressive polynomials; macoefs, the coefficients of the product of the moving average polynomials.
        """
        arcoefs = model.arparams()
        if add and model.order[1]> 0:
            arcoefs = np.insert(-arcoefs, 0, 1)
            for _ in range(model.order[1]):
                arcoefs = np.concatenate((np.array([arcoefs[0]]), np.diff(arcoefs), np.array([-arcoefs[-1]])))
            arcoefs = -arcoefs[1:]
        
        if add and model.seasonal_order[1] > 0:
            arcoefs = np.insert(-arcoefs, 0, 1)
            tmp = np.concatenate(np.zeros(model.seasonal_order[3] - 1), arcoefs, np.zeros(model.seasonal_order[3] - 1))
            tmp = tmp[model.seasonal_order[3]:] - tmp[:model.seasonal_order[3]]
            arcoefs = np.concatenate((np.array([arcoefs[0]]), tmp, np.array([-arcoefs[-1]])))
            if model.seasonal_order[1] == 2:
                tmp = np.concatenate(np.zeros(model.seasonal_order[3] - 1), arcoefs, np.zeros(model.seasonal_order[3] - 1))
                tmp = tmp[model.seasonal_order[3]:] - tmp[:model.seasonal_order[3]]
                arcoefs = np.concatenate((np.array([arcoefs[0]]), tmp, np.array([-arcoefs[-1]])))
            elif model.seasonal_order[1] > 2:
                raise Exception(f'Unsupported model seasonal difference D {model.seasonal_order[1]} > 2.')
            arcoefs = -arcoefs[1:]
        
        macoefs = model.maparams()
        macoefs = macoefs[np.arange(model.order[2] + model.seasonal_order[2] * model.seasonal_order[3])]

        return {
            'arcoefs': arcoefs,
            'macoefs': macoefs
        }

    @staticmethod
    def arima2poly(model: ARIMA, ar, ma, seasonal_ar=None, seasonal_ma=None, s=1):
        """
        Convert ARIMA model coefficients to polynomial form.

        Parameters:
        -----------
        ar : array_like
            Autoregressive coefficients.
        ma : array_like
            Moving average coefficients.
        seasonal_ar : array_like, optional
            Seasonal autoregressive coefficients. Default is None.
        seasonal_ma : array_like, optional
            Seasonal moving average coefficients. Default is None.
        s : int, optional
            Seasonal period. Default is 1.

        Returns:
        --------
        poly : ndarray
            Coefficients of the polynomial in increasing order of degree.
        """
        ar = model.arparams()
        ma = model.maparams()

        order_p = model.order[0]
        order_d = model.order[1]
        order_q = model.order[2]

        seasonal_order_P = model.seasonal_order[0]
        seasonal_order_D = model.seasonal_order[1]
        seasonal_order_Q = model.seasonal_order[2]
        seasonal_order_s = model.seasonal_order[3]

        if order_d > 0:
            # Create coefficients of the differencing polynomial
            pd = np.concatenate(([1], np.zeros(order_d - 1), [-1]))

            # Calculate polynomial representation of differenced ARIMA process
            ar = np.polymul(ar, pd)
            ma = np.polymul(ma, pd)

        # If there is a seasonal component, include it
        if seasonal_order_P > 0 or seasonal_order_Q > 0:
            poly = np.zeros(max(order_p, order_q, seasonal_order_P * seasonal_order_s, seasonal_order_Q * seasonal_order_s) + 1)
        else:
            poly = np.zeros(max(order_p, order_q) + 1)

        # Set the coefficients corresponding to the AR and MA terms
        poly[0] = 1
        poly[1:(order_p+1)] = -ar
        poly[1:(order_q+1)] += ma

        # If there is a seasonal component, set the corresponding coefficients
        if seasonal_order_P > 0:
            for i in range(seasonal_order_s, seasonal_order_P * seasonal_order_s + 1, seasonal_order_s):
                poly[i+1:i+q+1] += seasonal_ar[:q]
                seasonal_ar = seasonal_ar[q:]
        if seasonal_order_Q > 0:
            for i in range(s, seasonal_order_Q * s + 1, s):
                poly[i+1:i+p+1] -= seasonal_ma[:p]
                seasonal_ma = seasonal_ma[p:]

        return poly


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
            id0resid = list(range(1, tmp+1))
        else:
            id0resid = [1, 2]

        iter = 0
        while iter < self.maxit_oloop:
            
        
    def _fit(self):
        """
        """
        if self.tsmethod == 'AutoARIMA':
            self.ts_model = AutoARIMA(**self.args_tsmethod).fit(y=self.endog, X=self.exog).model_
        else:
            self.ts_model = ARIMA(**self.args_tsmethod).fit(y=self.endog, X=self.exog)

        return

    def fit(self):
        """
        """
        cval0 = self.cval
        res0 = res = self._fit()
        
        return

import numpy as np

def arima2poly(ar, ma, d, D, seasonal_ar=None, seasonal_ma=None, period=1, include_constant=True):
    """
    Convert ARIMA model coefficients to polynomial form.
    """
    p, d, q = len(ar), d, len(ma)
    if seasonal_ar is not None:
        P, D, Q, s = len(seasonal_ar), D, len(seasonal_ma), period
    else:
        P, D, Q, s = 0, 0, 0, 1
    poly_ar = np.r_[1, -ar]
    poly_ma = np.r_[1, ma]
    poly_seasonal_ar = np.r_[1]
    if seasonal_ar is not None:
        poly_seasonal_ar = np.r_[1, -seasonal_ar]
    poly_seasonal_ma = np.r_[1]
    if seasonal_ma is not None:
        poly_seasonal_ma = np.r_[1, seasonal_ma]
    poly = 1
    for k in range(D):
        poly = np.polymul(poly, np.r_[1, np.zeros(s-1), 1])
    poly = np.polymul(poly, poly_seasonal_ar)
    for k in range(d):
        poly = np.polymul(poly, np.r_[1, -1])
    poly = np.polymul(poly, poly_ar)
    print(poly)
    for k in range(q):
        poly = np.polymul(poly, np.r_[1, 0])
    poly = np.polymul(poly, poly_ma)
    for k in range(Q):
        poly = np.polymul(poly, np.r_[1, np.zeros(s-1), 1])
    poly = np.polymul(poly, poly_seasonal_ma)
    if include_constant:
        poly = np.polymul(poly, np.r_[1, np.zeros(s-1)]) + 1
    return poly
