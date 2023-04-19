"""
This module contains models for trend predictor for time series.
"""
from functools import reduce
import timeit
from copy import deepcopy
from itertools import product
from math import ceil, floor
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import torch.nn as nn
from torch import no_grad
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from ..algorithm.online_update import OnlineUpdateRule
from ..utility import PositiveFlt, PositiveInt, Prob
from .util import ModelPerformance, ModelPerformanceOnline, TimeSeriesData, TimeSeriesDataCollection


def run_epoch(model: nn.Module, optimizer: Any, loss_func: Any, data_loader: DataLoader, is_train: bool, clip_grad: Optional[PositiveFlt]=None):
    """ Run one epoch of training or evaluation.

    :param model: The model to be trained or tested.
    :param optimizer: The optimizer.
    :param loss_func: The loss function.
    :param data_loader: The data loader.
    :param is_train: Whether to train the model.
    :param clip_grad: The maximum norm of the gradient.
    :return: The average loss and the residuals.
    """
    loss_sum = 0
    total_samples = 0
    curr_resids = []
    for batch in data_loader:
        optimizer.zero_grad()

        data, label, index, dataset_index = batch ## Tensor, Tensor, Tensor, None/int
        data = [x.to(model.device) for x in data]
        label = label.to(model.device)
        pred = model(*data) ## Shape: (batch_size, n_ahead, n_endog), could be multiple tensors

        loss = loss_func(pred, label)
        resid = label - pred
        curr_resid = pd.DataFrame(resid.reshape(-1).detach().cpu().numpy(), columns=['residual']).assign(dataset_index=dataset_index)
        curr_resid['index'] = np.repeat(index.reshape(-1).cpu().numpy(), pred.shape[2]) ## batch
        curr_resid['n_ahead'] = np.tile(np.repeat(np.arange(pred.shape[1]), pred.shape[2]), pred.shape[0]) ## n_ahead
        curr_resid['n_endog'] = np.tile(np.arange(pred.shape[2]), pred.shape[0] * pred.shape[1]) ## n_endog
        curr_resids.append(curr_resid)

        if is_train:
            loss.backward()
            if clip_grad is not None:
                clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        if loss_func.reduction == 'sum':
            loss_sum += loss.item()
        else:
            loss_sum += loss.item() * index.size(0)
        total_samples += index.size(0)

    return loss_sum / total_samples, pd.concat(curr_resids, axis=0, ignore_index=True).drop_duplicates(keep='last')

def get_train_valid_test_indices(dataset: Union[TimeSeriesData, TimeSeriesDataCollection], test_ratio: Prob, n_fold: PositiveInt=3):
    """ Get train, validation and test indices lookup for each dataset and each fold.

    :param dataset: The dataset.
    :param test_ratio: The ratio of the test set.
    :param n_fold: The number of folds for cross validation, the K+1th fold will be treated as test set, Kth fold will be treated as validation set, and the first K-1 fold will be treated as dataset. n_fold has be at least 2.
    :return: The train, validation and test indices lookup for each dataset and each fold.
    """
    def translate_local_indice_to_global(start_indices, indice, dataset_index):
        if np.isnan(indice):
            return indice
        if isinstance(dataset, TimeSeriesData):
            return indice
        else:
            return indice + start_indices[int(dataset_index)]

    if isinstance(dataset, TimeSeriesData):
        datasets = [dataset]
    else:
        datasets = dataset.datasets

    dataset_start_indices = np.cumsum([0] + [len(x) for x in datasets[:-1]])

    train_indices_lookup = {} ## [fold_idx] -> List[int]
    valid_indices_lookup = {} ## [fold_idx] -> List[int]
    test_indices_lookup = [] ## [] -> List[int]
    for dataset_index, data in enumerate(datasets):
        test_size = ceil(len(data) * test_ratio)
        train_size = len(data) - test_size

        fold_size = floor(train_size / n_fold)
        for fold_idx in range(1, n_fold+1):
            train_start = 0 + (fold_idx - 1) * fold_size
            train_end = train_start + fold_size
            if fold_idx == n_fold:
                valid_start = np.nan
                valid_end = np.nan
            else:
                valid_start = train_end
                valid_end = valid_start + fold_size

            train_start = translate_local_indice_to_global(dataset_start_indices, train_start, dataset_index)
            train_end = translate_local_indice_to_global(dataset_start_indices, train_end, dataset_index)
            if fold_idx in train_indices_lookup:
                if not np.isnan(train_start) and not np.isnan(train_end):
                    train_indices_lookup[fold_idx] += list(range(train_start, train_end))
            else:
                if not np.isnan(train_start) and not np.isnan(train_end):
                    train_indices_lookup[fold_idx] = list(range(train_start, train_end))
                else:
                    train_indices_lookup[fold_idx] = []

            valid_start = translate_local_indice_to_global(dataset_start_indices, valid_start, dataset_index)
            valid_end = translate_local_indice_to_global(dataset_start_indices, valid_end, dataset_index)
            if fold_idx in valid_indices_lookup:
                if not np.isnan(valid_start) and not np.isnan(valid_end):
                    valid_indices_lookup[fold_idx] += list(range(valid_start, valid_end))
            else:
                if not np.isnan(valid_start) and not np.isnan(valid_end):
                    valid_indices_lookup[fold_idx] = list(range(valid_start, valid_end))
                else:
                    valid_indices_lookup[fold_idx] = []

        test_start = len(data) - test_size
        test_end = len(data)
        test_start = translate_local_indice_to_global(dataset_start_indices, test_start, dataset_index)
        test_end = translate_local_indice_to_global(dataset_start_indices, test_end, dataset_index)
        test_indices_lookup += list(range(test_start, test_end))
    return train_indices_lookup, valid_indices_lookup, test_indices_lookup

def train_and_evaluate_model(model: nn.Module,
                             optimizer: Any,
                             loss_func: Any,
                             dataset: Optional[Union[TimeSeriesData, TimeSeriesDataCollection]],
                             epochs: PositiveInt,
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
    :param epochs: The number of epochs.
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

    train_indices_lookup, valid_indices_lookup, test_indices_lookup = get_train_valid_test_indices(dataset, test_ratio, n_fold)

    non_empty_train_indices = any([len(x) > 0 for x in train_indices_lookup.values()])
    if non_empty_train_indices:
        train_start_time = timeit.default_timer()

        init_state_dict = deepcopy(model.state_dict())
        train_record = []
        train_valid_loss = []
        train_losses = []
        for fold in tqdm(train_indices_lookup, desc='Multi-fold validation', position=0, leave=True, total=len(train_indices_lookup)):
            model.load_state_dict(init_state_dict) ## Reset the model parameters
            train_indices = train_indices_lookup[fold]
            train_set = Subset(dataset, train_indices)
            train_loader = DataLoader(train_set, **dataloader_args)
            valid_indices = valid_indices_lookup[fold]
            if len(valid_indices) > 0:
                valid_set = Subset(dataset, valid_indices)
                valid_loader = DataLoader(valid_set, **dataloader_args)
            else:
                valid_loader = None

            with tqdm(total=epochs, desc='Epoch', position=1, leave=True) as pbar:
                for epoch in range(epochs):
                    train_loss, train_resid = run_epoch(model, optimizer, loss_func, train_loader, is_train=True, clip_grad=clip_grad)
                    if valid_loader is not None:
                        with no_grad():
                            valid_loss, _ = run_epoch(model, optimizer, loss_func, valid_loader, is_train=False, clip_grad=clip_grad)
                    curr_record = pd.Series({
                        'fold': fold,
                        'epoch': epoch,
                        'train_size': len(train_indices),
                        'valid_size': len(valid_indices),
                        'training_loss': train_loss,
                        'validation_loss': valid_loss if valid_loader is not None else np.nan
                    })
                    train_record.append(curr_record)
                    pbar.update(1)
                pbar.set_postfix(curr_record.to_dict())

            train_valid_loss.append(curr_record['validation_loss'])
            train_losses.append(curr_record['training_loss'])

        train_record = pd.concat([x.to_frame().T for x in train_record], ignore_index=True)
        train_stop_time = timeit.default_timer()

    ## Test the model
    non_empty_test_indices = len(test_indices_lookup) > 0
    if non_empty_test_indices:
        test_start_time = timeit.default_timer()
        test_set = Subset(dataset, test_indices_lookup)
        test_loader = DataLoader(test_set, **dataloader_args)
        with no_grad():
            test_loss, test_resid = run_epoch(model, optimizer, loss_func, test_loader, is_train=False, clip_grad=clip_grad)
        test_stop_time = timeit.default_timer()

    print(f'Averaged validation loss: {np.nanmean(train_valid_loss)}. Test loss: {test_loss}.')

    training_info = {'train_size': len(train_indices_lookup[n_fold]) if non_empty_train_indices else 0,
                     'train_loss_func': str(loss_func),
                     'train_optimizer': str(optimizer),
                     'train_clip_grad': clip_grad,
                     'train_epochs': epochs,
                     'train_start_time': train_start_time if non_empty_train_indices else None,
                     'train_stop_time': train_stop_time if non_empty_train_indices else None,
                     'train_elapsed_time': train_stop_time - train_start_time if non_empty_train_indices else None,
                     'train_n_fold': n_fold if non_empty_train_indices else None,
                     'average_train_loss': np.nanmean(train_losses) if non_empty_train_indices else None,
                     'last_train_loss': train_losses[-1] if non_empty_train_indices else None,
                     'average_validation_loss': np.nanmean(train_valid_loss) if non_empty_train_indices else None}
    training_info.update(dataloader_args)
    testing_info = {'test_size': len(test_indices_lookup),
                    'test_loss_func': str(loss_func),
                    'test_loss': test_loss if non_empty_test_indices else None,
                    'test_start_time': test_start_time if non_empty_test_indices else None,
                    'test_stop_time': test_stop_time if non_empty_test_indices else None,
                    'test_elapsed_time': test_stop_time - test_start_time if non_empty_test_indices else None}

    return ModelPerformance(model=model,
                            dataset=dataset,
                            training_record=train_record if non_empty_train_indices else pd.DataFrame(),
                            training_residuals=train_resid if non_empty_train_indices else pd.DataFrame(),
                            testing_residuals=test_resid if non_empty_test_indices else pd.DataFrame(),
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
                train_loss, train_resid = run_epoch(model, optimizer, loss_func, train_loader, is_train=True, clip_grad=clip_grad)
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
        test_loss, test_resid = run_epoch(model, optimizer, loss_func, test_loader, is_train=False) ## We simulate to observe one step at a time and predict one step at a time
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
