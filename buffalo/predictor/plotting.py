"""
This module contains functions for plotting metrics related to models.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_training_records(training_records: pd.DataFrame):
    """Plot the training records.

    :param training_records: The training records.
    """
    plt1 = sns.lineplot(x='epoch', y='training_loss', hue='fold', data=training_records)
    plt1.set_title('Training Loss over Time')
    plt1.set_xlabel('Epoch')
    plt1.set_ylabel('Training Loss')
    plt2 = sns.lineplot(x='epoch', y='validation_loss', hue='fold', data=training_records)
    plt2.set_title('Validation Loss over Time')
    plt2.set_xlabel('Epoch')
    plt2.set_ylabel('Validation Loss')
    return plt1, plt2
