"""
This module contains models for trend predictor for time series.
"""

from copy import deepcopy
from math import ceil, floor
from typing import Any, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from ..utility import (NonnegativeInt, PositiveFlt, PositiveInt, Prob,
                       create_parent_directory)


def train_model(model: nn.Module,
                optimizer: Any,
                loss_func: Any,
                trainset: Optional[Dataset],
                validation_ratio: Prob,
                epochs: PositiveInt,
                clip_grad: Optional[PositiveFlt]=None,
                multi_fold_valiation: bool=False,
                save_model: bool=False,
                save_path: Optional[str]=None,
                **dataloader_args) -> pd.DataFrame:
    """
    Train the model.

    :param model: The model to be trained.
    :param optimizer: The optimizer.
    :param loss_func: The loss function.
    :param trainset: The data set used to split the training set and validation set.
    :param validation_ratio: The ratio of validation set.
    :param epochs: The number of epochs.
    :param clip_grad: The maximum gradient norm to be clipped. If None, then no gradient clipping is performed.
    :param multi_fold_valiation: Whether to use multifold validation. If false and the validation ratio is positive, then only one validation set is used.
    :param save_model: Whether to save the trained model to file.
    :param save_path: Which filepath to save the trained model.
    :param dataloader_args: The arguments for the data loader.
    :return: The training record.
    """
    if trainset is None:
        trainset_not_provided = True
        trainset = model.train_set
    else:
        trainset_not_provided = False
    all_indices = set(range(len(trainset)))
    if validation_ratio == 0:
        train_indices = [all_indices]
    else:
        fold_size = floor(len(trainset) * validation_ratio)
        if multi_fold_valiation:
            n_folds = ceil(1 / validation_ratio)
            train_indices = [all_indices - set(range(i * fold_size, min((i + 1) * fold_size, len(trainset) - 1))) for i in range(n_folds)]
        else:
            n_folds = 1
            train_indices = [all_indices - set(range(0, min(fold_size, len(trainset) - 1)))]

    init_state_dict = deepcopy(model.state_dict())
    train_record = []
    train_resid = []
    train_model_params = []
    train_valid_loss = []
    for fold, train_indice in tqdm(enumerate(train_indices), desc='Multi-fold validation', position=0, leave=True, total=len(train_indices)):
        valid_indice = all_indices - train_indice
        model.load_state_dict(init_state_dict) ## Reset the model parameters
        with tqdm(total=epochs, desc='Epoch', position=1, leave=True) as pbar:
            for epoch in range(epochs):
                curr_resid = pd.Series(dtype='float32') ## Initalize the residuals to be nan
                t_loss_sum = 0
                v_loss_sum = 0

                train_set = Subset(trainset, list(train_indice))
                valid_set = Subset(trainset, list(valid_indice))
                train_loader = DataLoader(train_set, **dataloader_args)
                valid_loader = DataLoader(valid_set, **dataloader_args)

                ## Train Procedure
                for batch in train_loader:
                    optimizer.zero_grad()

                    data, label, index = batch
                    data = data.to(model.device)
                    label = label.to(model.device)
                    outputs = model(data)
                    pred = outputs

                    loss = loss_func(pred, label)
                    curr_resid = (pd.Series((label - pred).squeeze().detach().cpu().numpy(), index=index.squeeze().cpu().numpy())).combine_first(curr_resid)
                    loss.backward()
                    if clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    optimizer.step()
                    t_loss_sum += loss.item()

                if len(valid_set) > 0:
                    with torch.no_grad():
                        for batch in valid_loader:
                            data, label, index = batch
                            data = data.to(model.device)
                            label = label.to(model.device)
                            pred = model(data)

                            loss = loss_func(pred, label)
                            v_loss_sum += loss.item()

                curr_record = pd.Series({
                    'fold': fold,
                    'valid_start': min(valid_indice),
                    'valid_end': max(valid_indice),
                    'epoch': epoch,
                    'training_loss': t_loss_sum / len(train_loader),
                    'validation_loss': v_loss_sum / len(valid_loader)
                })
                train_record.append(curr_record)
                pbar.update(1)

            pbar.set_postfix(curr_record.to_dict())
        train_resid.append(curr_resid) ## Only append the residuals from the last epoch
        train_valid_loss.append(curr_record['validation_loss'])
        train_model_params.append(deepcopy(model.state_dict()))

    ## Find the best validation loss
    best_fold = train_valid_loss.index(min(train_valid_loss))
    model.load_state_dict(train_model_params[best_fold])
    train_record = pd.concat([x.to_frame().T for x in train_record], ignore_index=True)
    update_training_records(model, train_record, Subset(trainset, list(train_indices[best_fold])) if not trainset_not_provided else Dataset(), train_resid[best_fold], trainset_not_provided)
    if save_model:
        create_parent_directory(save_path)
        torch.save(model, save_path)
    print(f'Averaged validation loss: {np.mean(train_valid_loss)}. Best validation loss: {min(train_valid_loss)}')
    return train_record

def test_model(model: nn.Module, testset: Dataset, loss_func: Any, **dataloader_args):
    """
    Test the model.

    :param model: The model.
    :param data_loader: The test set DataLoader.
    :return: The test residuals.
    """
    test_data_loader = DataLoader(testset, **dataloader_args)
    test_loss = 0
    test_resid = pd.Series(dtype='float32')
    for batch in tqdm(test_data_loader, desc='Batched testing'):
        data, label, index = batch
        data = data.to(model.device)
        label = label.to(model.device)
        pred = model(data)
        test_resid = test_resid.combine_first(pd.Series((label - pred).squeeze().detach().cpu().numpy(), index=index.squeeze().cpu().numpy()))
        loss = loss_func(pred, label)
        test_loss += loss.item()
    print(f'Test loss: {test_loss / len(test_data_loader)}')
    return test_resid

def update_training_records(model: nn.Module, train_record: pd.DataFrame, train_set: Dataset, train_resid: pd.Series, append: bool=True):
    """
    Update the training records of the model.

    :param model: The model where training info are stored to.
    :param train_record: The training record generated by train_model function.
    :param train_set: The training set used by train_model function.
    :param train_resid: The training residuals generated by train_model function.
    :param append: If True, append the training info to the existing training info. Default: True.
    """
    if append:
        model.train_set = ConcatDataset([model.train_set, train_set])
        max_epoch = model.train_record['epoch'].max() + 1
        train_record['epoch'] = train_record['epoch'] - train_record['epoch'].min() + max_epoch
        model.train_record = pd.concat([model.train_record, train_record], axis=0)
        model.train_resid = model.train_resid.combine_first(train_resid)
    else:
        model.train_set = train_set
        model.train_record = train_record
        model.train_resid = train_resid


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
        self.train_set = Dataset()
        self.train_record = pd.DataFrame()
        self.train_resid = pd.Series(dtype='float32')
        self.info = {'name': 'RNN', 'input_size': input_size, 'hidden_size': hidden_size, 'output_size': output_size, 'num_layers': num_layers, 'nonlinearity': nonlinearity, 'bias': bias, 'dropout': dropout, 'bidirectional': bidirectional, 'str_rep': str(self)}

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
        self.info = {'name': 'LSTM', 'input_size': input_size, 'hidden_size': hidden_size, 'output_size': output_size, 'num_layers': num_layers, 'bias': bias, 'dropout': dropout, 'bidirectional': bidirectional, 'str_rep': str(self)}

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
