"""
This module contains the pipeline classes for the Buffalo project.
"""

from ..predictor import TimeSeriesData, models


class Canary:
    """
    Canary pipleline is a class that contains a data module and a outlier detection module.
    """

    def __init__(self) -> None:
        pass

    def run(self, dataset: TimeSeriesData, model: models.nn.Module) -> None:
        """
        Run the Canary pipeline.
        """
        pass
