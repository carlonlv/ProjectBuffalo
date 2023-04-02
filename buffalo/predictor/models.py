"""
This module contains models for trend predictor for time series.
"""

import copy
from math import ceil, floor
from typing import Any, Literal, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..utility import PositiveInt, Prob, concat_dict


class RNN(nn.Module):
    """
    This class provide wrapper for self defined RNN module.
    """

    def __init__(self,
                 input_size: PositiveInt,
                 hidden_size: PositiveInt,
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
        self.batch_norm = nn.BatchNorm1d(num_features=input_size).to(self.device)
        self.model = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity=nonlinearity, bias=bias, batch_first=True, dropout=dropout, bidirectional=bidirectional).to(self.device).to(self.device)

    def forward(self, input_v: torch.Tensor, h_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of input data and hidden state from previous cell.

        Let N = batch size, L = sequence length, D = 2 if bidirectional otherwise 1, H_in = input size, H_out = output size.

        :param input_v: tensor of shape (L, H_in) for unbatched input, or (N, L, H_in) when batched, containing the features of the input sequence. The input can also be a packed variable length sequence.
        :param h_0: tensor of shape (D * num_layers, H_out) for unbatched output or (D * num_layers, N, H_out) containing the initial hidden state for the input sequence batch. Defaults to zeros if not provided.
        """
        h_in = input_v.shape[-1]
        seq_len = input_v.shape[-2]
        if len(input_v.shape) == 2:
            ## Unbatched
            batch_num = 1
        else:
            batch_num = input_v.shape[0]
        input_v = input_v.view(batch_num, h_in * seq_len) ## Per time, per series batch normalization
        input_v = self.batch_norm(input_v)
        input_v = input_v.view(batch_num, seq_len, h_in)
        if batch_num == 1:
            input_v = input_v.squeeze(0)
        return self.model(input=input_v, h_0=h_0)

    def fit(self, opt: Any, loss: Any, data_loader: DataLoader, validation_ratio: Prob, epochs: PositiveInt, multi_fold_valiation: bool=False, verbose: bool=True):
        """
        Train the underlying model.

        :param opt: The optimizer.
        :param loss: The loss function.
        :param data_loader: The data set DataLoader used to split the training set and validation set.
        :param validation_ratio: The ratio of validation set.
        :param epochs: The number of epochs.
        """
        all_indices = set(range(len(data_loader.dataset)))
        if validation_ratio == 0:
            train_indices = [all_indices]
        else:
            fold_size = floor(len(data_loader.dataset) * validation_ratio)
            if multi_fold_valiation:
                n_folds = ceil(1 / validation_ratio)
                train_indices = [all_indices - set(range(i * fold_size, min((i + 1) * fold_size, len(data_loader.dataset) - 1))) for i in range(n_folds)]
            else:
                n_folds = 1
                train_indices = [all_indices - set(range(i * fold_size, min((i + 1) * fold_size, len(data_loader.dataset) - 1))) for i in range(n_folds)]

        train_record = []
        for train_indice in tqdm(train_indices):
            for epoch in tqdm(range(epochs)):
                t_loss_sum = 0
                v_loss_sum = 0

                train_set = Subset(data_loader.dataset, train_indice)
                valid_set = Subset(data_loader.dataset, all_indices - train_indice)
                train_loader = copy.copy(data_loader)
                train_loader.dataset = train_set
                valid_loader = copy.copy(data_loader)
                valid_loader.dataset = valid_set

                ## Train Procedure
                for batch in data_loader:
                    opt.zero_grad()

                    data, label, lens = batch
                    data, label = data.to(self.device), label.to(self.device)
                    pred = self.model(data, lens)

                    loss = loss(pred, label)
                    loss.backward()
                    opt.step()
                    t_loss_sum += loss.item()

                if len(valid_set) > 0:
                    with torch.no_grad():
                        for batch in valid_loader:
                            data, label, lens = batch
                            data, label = data.to(self.device), label.to(self.device)
                            pred = self.model(data, lens)

                            loss = loss(pred, label)
                            v_loss_sum += loss.item()

                curr_record = pd.Series({
                    'train_start': min(train_indice),
                    'train_end': max(train_indice),
                    'epoch': epoch,
                    'training_loss': t_loss_sum,
                    'validation_loss': v_loss_sum
                })
                train_record.append(curr_record)

                if verbose and (epoch % 10 == 0):
                    print(concat_dict(curr_record.to_dict()))

        return train_record
