"""
This module contains models for trend predictor for time series.
"""

from typing import Literal, Tuple, Optional, Any

import pandas as pd
import torch
import torch.nn as nn

from ..utility import PositiveInt, Prob


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
                 batch_first: bool=False,
                 dropout: Prob=0,
                 bidirectional: bool=False,
                 use_gpu: bool=True) -> None:
        """
        Initializer and configuration for RNN Autoencoder.

        :param input_size: The number of expected features in the input x.
        :param hidden_size: The number of features in the hidden state h.
        :param num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1.
        :param nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'.
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True.
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False.
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0.
        :param bidirectional: If True, becomes a bidirectional RNN. Default: False.
        :param use_gpu: If True, attempt to use cuda if GPU is available. Default: True.
        """
        super().__init__()
        if use_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.model = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity=nonlinearity, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional).to(self.device)

    def forward(self, input_v: torch.Tensor, h_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of input data and hidden state from previous cell.

        Let N = batch size, L = sequence length, D = 2 if bidirectional otherwise 1, H_in = input size, H_out = output size.

        :param input_v: tensor of shape (L, H_in) for unbatched input, (L, N, H_in) when batch_first=False or (N, L, H_in) where batch_first=True containing the features of the input sequence. The input can also be a packed variable length sequence.
        :param h_0: tensor of shape (D * num_layers, H_out) for unbatched output or (D * num_layers, N, H_out) containing the initial hidden state for the input sequence batch. Defaults to zeros if not provided.
        """
        input_v = input_v.to(self.device)
        return self.model(input=input_v, h_0=h_0)

    def train_loop(self, opt, t_loader, v_loader, epochs):
        """

        """
        for epoch in range(epochs):
            t_loss_sum = 0
            v_loss_sum = 0
            for batch in t_loader:
                opt.zero_grad()

                data, label, lens = batch
                data, label = data.to(self.device), label.to(self.device)
                pred = self.model(data, lens)

                loss = nn.CrossEntropyLoss()(pred, label)
                loss.backward()
                opt.step()
                t_loss_sum += loss.item()

            with torch.no_grad():
                for batch in v_loader:
                    data, label, lens = batch
                    data, label = data.to(self.device), label.to(self.device)
                    pred = self.model(data, lens)

                    loss = nn.CrossEntropyLoss()(pred, label)
                    v_loss_sum += loss.item()

            if epoch % 5 == 0:
                out = "Epoch {}: Train Loss {}, Val Loss {}"
                avg_t_loss = t_loss_sum / len(t_loader)
                avg_v_loss = v_loss_sum / len(v_loader)
                print(out.format(epoch, avg_t_loss, avg_v_loss))

    def fit(self, endog: pd.DataFrame, exog: Optional[pd.DataFrame], loss: Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Automatic Procedure for Detection of Outliers.

        :param endog: a training time series.
        :param exog: an optional matrix of regressors with the same number of rows as y.
        :param fit_args: Additional arguments besides endog and exog to be passed into fit() method.
        """

        return
