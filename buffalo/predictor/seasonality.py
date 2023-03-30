"""
Automatic procedure for detecting seasonality of time series.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.signal
import statsmodels.api as sm

from ..utility import PositiveFlt, PositiveInt, concat_list


class ChisquaredtestSeasonalityDetection:
    """
    Detect seasonality based on FFT and Chi-square tests of estimated power spectrum density. The significant frequency will be fitted using OLS.
    """
    def __init__(self,
                 alpha=0.05,
                 max_freq_nums: Optional[PositiveInt]=None,
                 max_period: Optional[PositiveInt]=None,
                 welch_args: Optional[Dict[str, Any]]=None) -> None:
        """
        Initializer for the automated seasonality detector.

        :param: alpha: The significance level to determine the significance of frequencies.
        :param max_freq_nums:
        """
        self.max_freq_nums = max_freq_nums
        self.max_period = max_period
        if welch_args is None:
            self.welch_args = {}
        else:
            self.welch_args = welch_args
        self.welch_args['scaling'] = 'spectrum'
        self.welch_args['return_onesided'] = True
        self.alpha = alpha

    def get_harmonic_exog(self, index_len: PositiveInt, freqs: List[PositiveFlt], index_offset: PositiveInt=0, add_const: bool=True):
        """
        Get Exogenous Harmonic Regressors.

        :param index_len: The length of current index.
        :param freqs: The frequencies to be added as regressors.
        :param index_offset: The index offset to start with, default 0.
        :param add_const: Whether to add vectors of 1s.
        :return: A dataframe with exogenous harmonic regressors.
        """
        exog = {}
        indices = np.arange(index_offset, index_offset + index_len)
        for freq in freqs:
            if isinstance(freq, float):
                exog[f'sin_{freq}'] = np.sin(2 * np.pi * freq * indices)
                exog[f'cos_{freq}'] = np.cos(2 * np.pi * freq * indices)
            else:
                if re.search(r'sin_\d+', freq):
                    ext_freq = float(freq.replace('sin_', ''))
                    exog[freq] = np.sin(2 * np.pi * ext_freq * indices)
                elif re.search(r'cos_\d+', freq):
                    ext_freq = float(freq.replace('cos_', ''))
                    exog[freq] = np.cos(2 * np.pi * ext_freq * indices)
        exog = pd.DataFrame(exog)
        if add_const:
            exog = sm.add_constant(exog, has_constant='add')
        return exog

    def fit(self, endog: pd.DataFrame) -> Dict[str, Tuple[List[PositiveFlt], Any]]:
        """
        Fit Harmonic Regression using identified most signifcant frequencies.

        :param endog: The endogenous time series. Each column represents a time series.
        :return: A dictionary with keys being the columns of endogenous.
        """
        result = {}
        for i in endog.columns:
            freq, psd = scipy.signal.welch(endog[i].to_numpy(), **self.welch_args)
            if self.max_period is not None:
                psd = psd[freq >= 1 / self.max_period]
                freq = freq[freq >= 1/ self.max_period]

            freq = freq[scipy.signal.find_peaks(psd)[0]]
            psd = psd[scipy.signal.find_peaks(psd)[0]]
            freq = freq[np.argsort(-psd)]
            psd = psd[np.argsort(-psd)]
            if self.max_freq_nums is not None:
                freq = freq[:self.max_freq_nums]
                psd = psd[:self.max_freq_nums]

            exog = self.get_harmonic_exog(endog.shape[0], freq)
            model = sm.OLS(endog[i].to_numpy(), exog.to_numpy(), missing='drop').fit()
            for _ in range(exog.shape[1]):
                pvalues = model.pvalues
                max_pvalue = pvalues.max()
                if max_pvalue > self.alpha:
                    index = pvalues.argmax()
                    exog = exog.drop(columns=exog.columns[index])
                    model = sm.OLS(endog[i].to_numpy(), exog.to_numpy(), missing='drop').fit()
                else:
                    break
            result[i] = (exog.columns, model)
            print(f'Detected seasonality {concat_list(exog.columns.drop("const"))} from {i}.')
        return result
