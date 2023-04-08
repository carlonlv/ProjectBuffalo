"""
This module contains models for trend predictor for time series.
"""

import timeit
from copy import deepcopy
from math import ceil, floor
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from ..utility import NonnegativeInt, PositiveFlt, PositiveInt, Prob
from .util import ModelPerformance, TimeSeriesData


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
    :param optimizer: The optimizer.
    :param loss_func: The loss function.
    :param dataset: The data set used to split the training, validation and test set.
    :param validation_ratio: The ratio of validation set.
    :param epochs_per_fold: The number of epochs per fold. Later folds will have more epochs because the model is trained on more data.
    :param n_fold: The number of folds for cross validation, the K+1th fold will be treated as test set, Kth fold will be treated as validation set, and the first K-1 fold will be treated as dataset. n_fold has be at least 2.
    :param clip_grad: The maximum gradient norm to be clipped. If None, then no gradient clipping is performed.
    :param save_record: Whether to save the trained model and trained record to file.
    :param save_path: Which filepath to save the trained model.
    :param dataloader_args: The arguments for the data loader.
    :return: The training record.
    """
    def run_epoch(data_loader: DataLoader, is_train: bool, residual_cols: List[str]):
        loss_sum = 0
        curr_resid = pd.DataFrame(columns=residual_cols) ## Initalize the residuals to be nan
        for batch in data_loader:
            optimizer.zero_grad()

            data, label, index = batch
            data = data.to(model.device)
            label = label.to(model.device)
            outputs = model(data)
            pred = outputs

            loss = loss_func(pred, label)
            curr_resid = (pd.DataFrame((label - pred).detach().cpu().numpy(), index=index.squeeze().cpu().numpy(), columns=residual_cols)).combine_first(curr_resid)

            if is_train:
                loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

            loss_sum += loss.item()

        return loss_sum / len(data_loader), curr_resid

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
        # train_resids = []
        # train_model_params = []
        # train_indices = []
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

                    train_loss, train_resid = run_epoch(train_loader, is_train=True, residual_cols=endog_cols)

                    if valid_loader is not None:
                        with torch.no_grad():
                            valid_loss, _ = run_epoch(valid_loader, is_train=False, residual_cols=endog_cols)

                    curr_record = pd.Series({
                        'fold': fold,
                        'epoch': epoch,
                        'train_start': train_indice[0],
                        'train_end': train_indice[1],
                        'valid_start': valid_indice[0] if valid_loader is not None else np.nan,
                        'valid_end': valid_indice[1] if valid_loader is not None else np.nan,
                        'training_loss': train_loss,
                        'validation_loss': valid_loss if valid_loader is not None else np.nan
                    })
                    train_record.append(curr_record)
                    pbar.update(1)
                pbar.set_postfix(curr_record.to_dict())

            train_valid_loss.append(curr_record['validation_loss'])
            # train_resids.append(train_resid) ## Only append the residuals from the last epoch
            # train_indices.append(train_indice)
            # train_model_params.append(deepcopy(model.state_dict()))

        # best_fold = train_valid_loss.index(min(train_valid_loss))
        # model.load_state_dict(train_model_params[best_fold])
        train_record = pd.concat([x.to_frame().T for x in train_record], ignore_index=True)
        train_stop_time = timeit.default_timer()

    ## Test the model
    if test_size > 0:
        test_start_time = timeit.default_timer()
        test_set = Subset(dataset, range(len(dataset)-test_size, len(dataset)))
        test_loader = DataLoader(test_set, **dataloader_args)
        with torch.no_grad():
            test_loss, test_resid = run_epoch(test_loader, is_train=False, residual_cols=endog_cols)
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
                            training_residuals=train_resid if train_size > 0 else pd.Series(dtype='float32'),
                            testing_residuals=test_resid if test_size > 0 else pd.Series(dtype='float32'),
                            training_info=training_info,
                            testing_info=testing_info)


class RNN(nn.Module):
    """
    This class provide wrapper for self defined RNN module.
    """

    def __init__(self,
                 input_size: PositiveInt,
                 hidden_size: PositiveInt,
                 output_size: PositiveInt,
                 num_layers: PositiveInt=1,
                 nonlinearity: Literal['tanh', 'relu']='tanh',
                 bias: bool=True,
                 dropout: Prob=0,
                 bidirectional: bool=False,
                 use_gpu: bool=True) -> None:
        """
        Initializer and configuration for RNN Autoencoder.

        :param input_size: The number of expected features in the input x, ie, number of features per time step.
        :param hidden_size: The number of features in the hidden state h.
        :param output_size: The number of features in the output.
        :param num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1.
        :param nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'.
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0.
        :param bidirectional: If True, becomes a bidirectional RNN. Default: False.
        :param use_gpu: If True, attempt to use cuda if GPU is available. Default: True.
        """
        super().__init__()
        if use_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = nn.BatchNorm1d(num_features=input_size).to(self.device)
        self.model = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity=nonlinearity, bias=bias, batch_first=True, dropout=dropout, bidirectional=bidirectional).to(self.device)
        self.f_c = nn.Linear(in_features=hidden_size*(2 if not bidirectional else 4), out_features=output_size).to(self.device)
        batchnorm_param_count = sum(p.numel() for p in self.batch_norm.parameters() if p.requires_grad)
        batchnorm_connection_count = sum([torch.prod(torch.tensor(param.shape)).item() for name, param in self.batch_norm.named_parameters() if 'weight' in name])
        rnn_param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        rnn_connection_count = sum([torch.prod(torch.tensor(param.shape)).item() for name, param in self.model.named_parameters() if 'weight' in name])
        fc_param_count = sum(p.numel() for p in self.f_c.parameters() if p.requires_grad)
        fc_connection_count = sum([torch.prod(torch.tensor(param.shape)).item() for name, param in self.f_c.named_parameters() if 'weight' in name])
        self.info = {'name': 'RNN',
                     'input_size': input_size,
                     'hidden_size': hidden_size,
                     'output_size': output_size,
                     'num_layers': num_layers,
                     'nonlinearity': nonlinearity,
                     'bias': bias,
                     'dropout': dropout,
                     'bidirectional': bidirectional,
                     'str_rep': str(self),
                     'param_count': batchnorm_param_count + rnn_param_count + fc_param_count,
                     'connection_count': batchnorm_connection_count + rnn_connection_count + fc_connection_count}

    def forward(self, input_v: torch.Tensor, h_0: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of input data and hidden state from previous cell.

        Let N = batch size, L = sequence length, D = 2 if bidirectional otherwise 1, H_in = input size, H_out = output size.

        :param input_v: tensor of shape (N, L, H_in) when batched, containing the features of the input sequence. The input can also be a packed variable length sequence.
        :param h_0: tensor of shape (D * num_layers, N, H_out) containing the initial hidden state for the input sequence batch. Defaults to zeros if not provided.
        :return: output: tensor of shape (N, L, D * H_out) when batched, containing the output features h_t from the last layer of the RNN, for each t. h_n: tensor of shape (N, D * num_layers, H_out) containing the hidden state for t = L.
        """
        batch_num = input_v.shape[0]
        input_v = input_v.reshape(-1, self.input_size)
        input_v = self.batch_norm(input_v) ## Per series batch normalization, across all samples in batch and all time steps
        input_v = input_v.reshape(batch_num, -1, self.input_size)
        output_v, hidden_v = self.model(input_v, h_0)
        output_v = output_v[:, -1, :]  # Select last output of each sequence
        hidden_v = hidden_v[-1] if not self.bidirectional else torch.cat((hidden_v[-1], hidden_v[-2]), dim=1)  # Select last layer hidden state
        output_v = torch.cat((output_v, hidden_v), dim=1)
        return self.f_c(output_v)


class LSTM(nn.Module):
    """
    This class provide wrapper for self defined RNN module.
    """

    def __init__(self,
                 input_size: PositiveInt,
                 hidden_size: PositiveInt,
                 output_size: PositiveInt,
                 num_layers: PositiveInt=1,
                 bias: bool=True,
                 dropout: Prob=0,
                 bidirectional: bool=False,
                 proj_size: NonnegativeInt=0,
                 use_gpu: bool=True) -> None:
        """
        Initializer and configuration for RNN Autoencoder.

        :param input_size: The number of expected features in the input x, ie, number of features per time step.
        :param hidden_size: The number of features in the hidden state h.
        :param output_size: The number of features in the output.
        :param num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1.
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0.
        :param bidirectional: If True, becomes a bidirectional RNN. Default: False.
        :param use_gpu: If True, attempt to use cuda if GPU is available. Default: True.
        """
        super().__init__()
        if use_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = nn.BatchNorm1d(num_features=input_size).to(self.device)
        self.model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=True, dropout=dropout, bidirectional=bidirectional, proj_size=proj_size).to(self.device)
        self.f_c = nn.Linear(in_features=(hidden_size+2*(proj_size if proj_size > 0 else hidden_size))*(2 if bidirectional else 1), out_features=output_size).to(self.device)
        batchnorm_param_count = sum(p.numel() for p in self.batch_norm.parameters() if p.requires_grad)
        batchnorm_connection_count = sum([torch.prod(torch.tensor(param.shape)).item() for name, param in self.batch_norm.named_parameters() if 'weight' in name])
        lstm_param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        lstm_connection_count = sum([torch.prod(torch.tensor(param.shape)).item() for name, param in self.model.named_parameters() if 'weight' in name])
        fc_param_count = sum(p.numel() for p in self.f_c.parameters() if p.requires_grad)
        fc_connection_count = sum([torch.prod(torch.tensor(param.shape)).item() for name, param in self.f_c.named_parameters() if 'weight' in name])
        self.info = {'name': 'LSTM',
                     'input_size': input_size,
                     'hidden_size': hidden_size,
                     'output_size': output_size,
                     'num_layers': num_layers,
                     'bias': bias,
                     'dropout': dropout,
                     'bidirectional': bidirectional,
                     'str_rep': str(self),
                     'param_count': batchnorm_param_count + lstm_param_count + fc_param_count,
                     'connection_count': batchnorm_connection_count + lstm_connection_count + fc_connection_count}

    def forward(self, input_v: torch.Tensor, h_0: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of input data and hidden state from previous cell.

        Let N = batch size, L = sequence length, D = 2 if bidirectional otherwise 1, H_in = input size, H_out = output size.

        :param input_v: tensor of shape (L, H_in) for unbatched input, or (N, L, H_in) when batched, containing the features of the input sequence. The input can also be a packed variable length sequence.
        :param h_0: tensor of shape (D * num_layers, H_out) for unbatched output or (D * num_layers, N, H_out) containing the initial hidden state for the input sequence batch. Defaults to zeros if (h_0, c_0) not provided.
        :param c_0: tensor of shape (D * num_layers, H_cell) for unbatched output or (D * num_layers, N, H_cell) containing the initial cell state for the input sequence batch. Defaults to zeros if (h_0, c_0) not provided.
        """
        batch_num = input_v.shape[0]
        input_v = input_v.reshape(-1, self.input_size)
        input_v = self.batch_norm(input_v) ## Per series batch normalization, across all samples in batch and all time steps
        input_v = input_v.reshape(batch_num, -1, self.input_size)
        output_v, (hidden_v, cell_v) = self.model(input_v, h_0)
        output_v = output_v[:, -1, :]  # Select last output of each sequence
        hidden_v = hidden_v[-1] if not self.bidirectional else torch.cat((hidden_v[-1], hidden_v[-2]), dim=1) # Select last hidden state
        cell_v = cell_v[-1] if not self.bidirectional else torch.cat((cell_v[-1], cell_v[-2]), dim=1) # Select last cell state
        output_v = torch.cat((output_v, hidden_v, cell_v), dim=1)
        return self.f_c(output_v)
