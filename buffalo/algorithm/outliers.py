"""
This module contains algorithms for identifying/removing/predicting outliers.
"""

from typing import Optional, List, Literal, Dict, Any

import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

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
        tsmethod: Literal["auto.arima", "arima"]='auto.arima',
        args_tsmethod: Optional[Dict[str, Any]]=None,
        logfile: Optional[str]=None,
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
        self.discard_cval = discard_cval
        self.tsmethod = tsmethod
        self.args_tsmethod = args_tsmethod
        self.logfile = logfile
        self.check_rank = check_rank
