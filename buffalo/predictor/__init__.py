"""
This module contains models for trend predictor for time series.
"""
import timeit
from copy import deepcopy
from itertools import product
from math import ceil, floor
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch.nn as nn
from torch import no_grad
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from ..algorithm.online_update import OnlineUpdateRule
from ..utility import PositiveFlt, PositiveInt, Prob
from .util import ModelPerformance, ModelPerformanceOnline, TimeSeriesData


def run_epoch(model: nn.Module, optimizer: Any, loss_func: Any, data_loader: DataLoader, is_train: bool, residual_cols: List[str], clip_grad: Optional[PositiveFlt]=None):
    """ Run one epoch of training or evaluation.

    :param model: The model to be trained or tested.
    :param optimizer: The optimizer.
    :param loss_func: The loss function.
    :param data_loader: The data loader.
    :param is_train: Whether to train the model.
    :param residual_cols: The columns of the residuals to be returned, typically the same as endog columns.
    :param clip_grad: The maximum norm of the gradient.
    :return: The average loss and the residuals.
    """
    loss_sum = 0
    total_samples = 0
    col_name = [f'{x}:{y}' for x, y in product(residual_cols, range(1, 1+model.n_ahead))]
    curr_resid = pd.DataFrame(columns=col_name) ## Initalize the residuals to be nan
    for batch in data_loader:
        optimizer.zero_grad()

        data, label, index = batch
        data = data.to(model.device)
        label = label.to(model.device)
        pred = model(data) ## Shape: (batch_size, n_ahead, n_endog)

        loss = loss_func(pred, label)
        resid = label - pred
        curr_resid = (pd.DataFrame(resid.reshape(resid.shape[0],-1).detach().cpu().numpy(), index=index[:,0].cpu().numpy(), columns=col_name)).combine_first(curr_resid)

        if is_train:
            loss.backward()
            if clip_grad is not None:
                clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        if loss_func.reduction == 'sum':
            loss_sum += loss.item()
        else:
            loss_sum += loss.item() * data.size(0)
        total_samples += data.size(0)

    return loss_sum / total_samples, curr_resid

def train_and_evaluate_model(model: nn.Module,
                             optimizer: Any,
                             loss_func: Any,
                             dataset: Optional[TimeSeriesData],
                             epochs_per_fold: PositiveInt,
                             test_ratio: Prob,
                             n_fold: PositiveInt=3,
                             clip_grad: Optional[PositiveFlt]=None,
                             **dataloader_args) -> ModelPerformance:
    """
    Train and Evaluate the model.

    :param model: The model to be trained.
    :param  : The optimizer.
    :param loss_func: The loss function.
    :param dataset: The data set used to split the training, validation and test set.
    :param epochs_per_fold: The number of epochs per fold. Later folds will have more epochs because the model is trained on more data.
    :param test_ratio: The ratio of the test set.
    :param n_fold: The number of folds for cross validation, the K+1th fold will be treated as test set, Kth fold will be treated as validation set, and the first K-1 fold will be treated as dataset. n_fold has be at least 2.
    :param clip_grad: The maximum gradient norm to be clipped. If None, then no gradient clipping is performed.
    :param save_record: Whether to save the trained model and trained record to file.
    :param save_path: Which filepath to save the trained model.
    :param dataloader_args: The arguments for the data loader.
    :return: The training record.
    """
    dataset_not_provided = dataset is None
    if dataset_not_provided:
        dataset = model.dataset

    test_size = ceil(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    endog_cols = dataset.endog.columns

    if train_size > 0:
        train_start_time = timeit.default_timer()
        assert n_fold > 0, 'n_fold must be at least 1.'

        if n_fold > 1:
            ## Cross validation, train, validation and test split
            fold_size = floor(train_size / n_fold)
            indices = [((0, i*fold_size), (i*fold_size, (i+1)*fold_size)) for i in range(1, n_fold)]
            indices.append(((0, train_size-1), ()))
        else:
            ## No cross validation, only train and test split
            indices = [((0, train_size-1), ())]

        init_state_dict = deepcopy(model.state_dict())
        train_record = []
        train_valid_loss = []
        for fold, (train_indice, valid_indice) in tqdm(enumerate(indices), desc='Multi-fold validation', position=0, leave=True, total=len(indices)):
            model.load_state_dict(init_state_dict) ## Reset the model parameters
            epochs = (fold + 1) * epochs_per_fold
            with tqdm(total=epochs, desc='Epoch', position=1, leave=True) as pbar:
                for epoch in range(epochs):
                    train_set = Subset(dataset, range(*train_indice))
                    train_loader = DataLoader(train_set, **dataloader_args)
                    if len(valid_indice) > 0 and valid_indice[1] > valid_indice[0]:
                        valid_set = Subset(dataset, range(*valid_indice))
                        valid_loader = DataLoader(valid_set, **dataloader_args)
                    else:
                        valid_loader = None

                    train_loss, train_resid = run_epoch(model, optimizer, loss_func, train_loader, is_train=True, residual_cols=endog_cols, clip_grad=clip_grad)

                    if valid_loader is not None:
                        with no_grad():
                            valid_loss, _ = run_epoch(model, optimizer, loss_func, valid_loader, is_train=False, residual_cols=endog_cols, clip_grad=clip_grad)

                    curr_record = pd.Series({
                        'fold': fold,
                        'epoch': epoch,
                        'train_start': train_indice[0],
                        'train_end': train_indice[1]-1,
                        'valid_start': valid_indice[0] if valid_loader is not None else np.nan,
                        'valid_end': valid_indice[1]-1 if valid_loader is not None else np.nan,
                        'training_loss': train_loss,
                        'validation_loss': valid_loss if valid_loader is not None else np.nan
                    })
                    train_record.append(curr_record)
                    pbar.update(1)
                pbar.set_postfix(curr_record.to_dict())

            train_valid_loss.append(curr_record['validation_loss'])

        train_record = pd.concat([x.to_frame().T for x in train_record], ignore_index=True)
        train_stop_time = timeit.default_timer()

    ## Test the model
    if test_size > 0:
        test_start_time = timeit.default_timer()
        test_set = Subset(dataset, range(len(dataset)-test_size, len(dataset)))
        test_loader = DataLoader(test_set, **dataloader_args)
        with no_grad():
            test_loss, test_resid = run_epoch(model, optimizer, loss_func, test_loader, is_train=False, residual_cols=endog_cols, clip_grad=clip_grad)
        test_stop_time = timeit.default_timer()

    print(f'Averaged validation loss: {np.nanmean(train_valid_loss)}. Test loss: {test_loss}.')

    training_info = {'train_start': 0 if train_size > 0 else None,
                     'train_end': train_size-1 if train_size > 0 else None,
                     'train_loss_func': str(loss_func),
                     'train_optimizer': str(optimizer),
                     'train_clip_grad': clip_grad,
                     'train_epochs': epochs,
                     'train_start_time': train_start_time if train_size > 0 else None,
                     'train_stop_time': train_stop_time if train_size > 0 else None,
                     'train_elapsed_time': train_stop_time - train_start_time if train_size > 0 else None,
                     'train_n_fold': n_fold if train_size > 0 else None,
                     'average_validation_loss': np.nanmean(train_valid_loss) if train_size > 0 else None}
    testing_info = {'test_start': len(dataset)-test_size if test_size > 0 else None,
                    'test_end': len(dataset) if test_size > 0 else None,
                    'test_loss_func': str(loss_func),
                    'test_loss': test_loss if test_size > 0 else None,
                    'test_start_time': test_start_time if test_size > 0 else None,
                    'test_stop_time': test_stop_time if test_size > 0 else None,
                    'test_elapsed_time': test_stop_time - test_start_time if test_size > 0 else None}

    return ModelPerformance(model=model,
                            dataset=dataset,
                            training_record=train_record if train_size > 0 else pd.DataFrame(),
                            training_residuals=train_resid if train_size > 0 else pd.DataFrame(),
                            testing_residuals=test_resid if test_size > 0 else pd.DataFrame(),
                            training_info=training_info,
                            testing_info=testing_info)


def train_and_evaluate_model_online(model: nn.Module,
                                    dataset: Optional[TimeSeriesData],
                                    update_rule: OnlineUpdateRule,
                                    optimizer: Any,
                                    loss_func: Any,
                                    train_ratio: Prob,
                                    **dataloader_args) -> ModelPerformance:
    """
    Train and Evaluate the model in an Online Fashion.

    :param model: The model to be trained, could be pretrained model.
    :param optimizer: The optimizer.
    :param loss_func: The loss function.
    :param dataset: The data set used to split the training and test set.
    :param clip_grad: The maximum gradient norm to be clipped. If None, then no gradient clipping is performed.
    :param save_record: Whether to save the trained model and trained record to file.
    :param save_path: Which filepath to save the trained model.
    :param dataloader_args: The arguments for the data loader.
    :return: The training record.
    """
    dataset_not_provided = dataset is None
    if dataset_not_provided:
        dataset = model.dataset

    start_index = ceil(len(dataset) * train_ratio)
    end_index = len(dataset)
    endog_cols = dataset.endog.columns

    update_rule.clear_logs()

    start_time = timeit.default_timer()
    for t_index in tqdm(range(start_index, end_index), desc='Online training and testing.', position=0, leave=True):
        ## Decide whether to train the model or not, assume tindex is already observed
        update_rule.collect_obs(Subset(dataset, range(t_index, t_index+1)))
        train_indices, epochs, clip_grad = update_rule.get_train_settings(t_index) ## Get the indices of the data to be used to train the model
        if len(train_indices) > 0:
            train_loader = DataLoader(Subset(dataset, train_indices), **dataloader_args)
            train_records = []
            for epoch in range(epochs):
                train_loss, train_resid = run_epoch(model, optimizer, loss_func, train_loader, is_train=True, residual_cols=endog_cols, clip_grad=clip_grad)
                curr_record = pd.DataFrame({
                    'fold': 0,
                    'epoch': epoch,
                    'train_start': min(train_indices),
                    'train_end': max(train_indices),
                    'training_loss': train_loss
                }, index=[t_index])
                train_records.append(curr_record)
            update_rule.collect_train_stats(t_index, train_loss, train_resid, pd.concat(train_records, axis=0))

        test_loader = DataLoader(Subset(dataset, range(t_index+1, t_index+dataset.label_len+1)), batch_size=1)
        test_loss, test_resid = run_epoch(model, optimizer, loss_func, test_loader, is_train=False, residual_cols=endog_cols) ## We simulate to observe one step at a time and predict one step at a time
        update_rule.collect_test_stats(t_index, test_loss, test_resid)
    stop_time = timeit.default_timer()

    info = {'loss_func': str(loss_func),
            'optimizer': str(optimizer),
            'start_time': start_time,
            'stop_time': stop_time,
            'elapsed_time': stop_time - start_time}

    return ModelPerformanceOnline(model=model,
                                  dataset=dataset,
                                  update_rule=update_rule,
                                  info=info)
