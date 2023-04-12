"""
This module contains models for trend predictor for time series.
"""


from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from ..utility import NonnegativeInt, PositiveInt, Prob


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
