"""
This module contains models for trend predictor for time series.
"""


from typing import Any, Callable, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..utility import NonnegativeInt, PositiveFlt, PositiveInt, Prob


class RNN(nn.Module):
    """
    This class provide wrapper for self defined RNN module.
    """

    def __init__(self,
                 input_size: PositiveInt,
                 hidden_size: PositiveInt,
                 output_size: PositiveInt,
                 n_ahead: PositiveInt=1,
                 softmax_ouput: bool=False,
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
        :param n_ahead: The number of time steps to predict. Default: 1.
        :param softmax_ouput: If True, the output will be passed through a softmax function. Default: False.
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
        self.n_ahead = n_ahead
        self.bidirectional = bidirectional
        self.batch_norm = nn.BatchNorm1d(num_features=input_size).to(self.device)
        self.model = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity=nonlinearity, bias=bias, batch_first=True, dropout=dropout, bidirectional=bidirectional).to(self.device)
        self.f_c = nn.Linear(in_features=hidden_size*(1 if not bidirectional else 2), out_features=output_size).to(self.device)
        self.f_c = nn.Sequential(self.f_c, nn.Softmax(dim=-1)) if softmax_ouput else self.f_c
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
                     'n_ahead': n_ahead,
                     'softmax_ouput': softmax_ouput,
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
        output_v, _ = self.model(input_v, h_0)
        output_v = output_v[:, -self.n_ahead:, :]  # Select last output of each sequence
        return self.f_c(output_v)


class LSTM(nn.Module):
    """
    This class provide wrapper for self defined RNN module.
    """

    def __init__(self,
                 input_size: PositiveInt,
                 hidden_size: PositiveInt,
                 output_size: PositiveInt,
                 n_ahead: PositiveInt=1,
                 softmax_ouput: bool=False,
                 num_layers: PositiveInt=1,
                 bias: bool=True,
                 dropout: Prob=0,
                 bidirectional: bool=False,
                 use_gpu: bool=True) -> None:
        """
        Initializer and configuration for RNN Autoencoder.

        :param input_size: The number of expected features in the input x, ie, number of features per time step.
        :param hidden_size: The number of features in the hidden state h.
        :param output_size: The number of features in the output.
        :param n_ahead: Number of time steps to predict. Default: 1.
        :param softmax_ouput: If True, apply softmax to output. Default: False.
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
        self.n_ahead = n_ahead
        self.bidirectional = bidirectional
        self.batch_norm = nn.BatchNorm1d(num_features=input_size).to(self.device)
        self.model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=True, dropout=dropout, bidirectional=bidirectional, proj_size=output_size).to(self.device)
        self.model = nn.Sequential(self.batch_norm, nn.Softmax(dim=-1)) if softmax_ouput else self.model
        batchnorm_param_count = sum(p.numel() for p in self.batch_norm.parameters() if p.requires_grad)
        batchnorm_connection_count = sum([torch.prod(torch.tensor(param.shape)).item() for name, param in self.batch_norm.named_parameters() if 'weight' in name])
        lstm_param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        lstm_connection_count = sum([torch.prod(torch.tensor(param.shape)).item() for name, param in self.model.named_parameters() if 'weight' in name])
        self.info = {'name': 'LSTM',
                     'input_size': input_size,
                     'hidden_size': hidden_size,
                     'output_size': output_size,
                     'n_ahead': n_ahead,
                     'softmax_ouput': softmax_ouput,
                     'num_layers': num_layers,
                     'bias': bias,
                     'dropout': dropout,
                     'bidirectional': bidirectional,
                     'str_rep': str(self),
                     'param_count': batchnorm_param_count + lstm_param_count,
                     'connection_count': batchnorm_connection_count + lstm_connection_count}

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
        output_v, (_, _) = self.model(input_v, h_0)
        output_v = output_v[:, -self.n_ahead:, :]  # Select last output of each sequence
        return output_v


class Transformer(nn.Module):
    """ Transformer class to model outlier probability.
    """

    def __init__(self,
                 d_model,
                 output_dimension,
                 n_ahead: PositiveInt = 1,
                 softmax_ouput: bool=False,
                 causal_encoder: bool=False,
                 nhead: PositiveInt=8,
                 num_encoder_layers: PositiveInt=6,
                 num_decoder_layers: PositiveInt=6,
                 dim_feedforward: PositiveInt=2048,
                 dropout: Prob=0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]='relu',
                 custom_encoder: Optional[Any]=None,
                 custom_decoder: Optional[Any]=None,
                 layer_norm_eps: PositiveFlt=1e-05,
                 norm_first: bool=False,
                 use_gpu: bool=True) -> None:
        """
        Initializer for the Transformer class. To use transformer or other custom encoder/decoder, one should configure TimeSeriesData class with target sequence and source sequence as endogenous series, other features will be treated as exogenous series. split_endog must be used to split endogenous series into target sequence and source sequence. target_indices should be set to correspond the indices of target sequence. Also label_len should be configured to be the same as seq_len.

        :param d_model: the number of expected features in the encoder/decoder inputs (default=512), embedding size.
        :param output_dimension: the number of expected features in the final inputs, embedding size.
        :param n_ahead: the number of future steps to predict (default=1).
        :param softmax_ouput: Whether to use softmax to output probability (default=False).
        :param causal_encoder: Whether to use causal encoder (default=False).
        :param nhead: the number of heads in the multiheadattention models (default=8).
        :param num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        :param num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        :param dim_feedforward: the dimension of the feedforward network model (default=2048).
        :param dropout: the dropout value (default=0.1).
        :param activation: the activation function of intermediate layer, relu or gelu (default=relu).
        :param custom_encoder: custom encoder (default=None).
        :param custom_decoder: custom decoder (default=None).
        :param layer_norm_eps: the eps value in layer normalization components (default=1e-05).
        :param norm_first: if True, normalization is done before each sub-layer (default=False).
        :param use_gpu: whether to use gpu (default=True).
        """
        super().__init__()
        if use_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.n_ahead = n_ahead
        self.causal_encoder = causal_encoder
        self.model = nn.Transformer(d_model=d_model,
                                    nhead=nhead,
                                    num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout,
                                    activation=activation,
                                    custom_encoder=custom_encoder,
                                    custom_decoder=custom_decoder,
                                    layer_norm_eps=layer_norm_eps,
                                    batch_first=True,
                                    norm_first=norm_first,
                                    device=self.device)
        self.f_c = nn.Sequential(nn.Linear(d_model, output_dimension), nn.Softmax(dim=-1) if softmax_ouput else nn.Identity())
        transformer_param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        transformer_connection_count = sum([torch.prod(torch.tensor(param.shape)).item() for name, param in self.model.named_parameters() if 'weight' in name])
        fc_param_count = sum(p.numel() for p in self.f_c.parameters() if p.requires_grad)
        fc_connection_count = sum([torch.prod(torch.tensor(param.shape)).item() for name, param in self.f_c.named_parameters() if 'weight' in name])
        self.info = {'name': 'Transformer',
                     'n_ahead': n_ahead,
                     'softmax_ouput': softmax_ouput,
                     'causal_encoder': causal_encoder,
                     'd_model': d_model,
                     'nhead': nhead,
                     'num_encoder_layers': num_encoder_layers,
                     'num_decoder_layers': num_decoder_layers,
                     'dim_feedforward': dim_feedforward,
                     'dropout': dropout,
                     'activation': activation,
                     'custom_encoder': str(custom_encoder),
                     'custom_decoder': str(custom_decoder),
                     'layer_norm_eps': layer_norm_eps,
                     'norm_first': norm_first,
                     'param_count': transformer_param_count + fc_param_count,
                     'connection_count': transformer_connection_count + fc_connection_count}

    def forward(self, src, tgt) -> torch.Tensor:
        """
        Forward propagation. Memory mask is not used, assume the target series depends on both the past and future of the source series.
        Causal masks are applied for both source and target series to prevent both series from looking into the future.
        src_key_padding_mask, tgt_key_padding_mask and memory_key_padding_mask are not used, assume no paddings are used to ensure the same length of all series. Padding may be used to mask time steps with missing values.

        :param src: the sequence to the encoder (required). (N, S, E) for batched.
        :param tgt: the sequence to the decoder (required). (N, T, E) for batched.
        :return: the output of the transformer.
        """
        src_causal_mask = self.model.generate_square_subsequent_mask(src.size(1)).to(self.device) if self.causal_encoder else None ## (S, S)
        tgt_causal_mask = self.model.generate_square_subsequent_mask(tgt.size(1)).to(self.device) ## (T, T)
        xvc = self.model(src=src,
                         tgt=tgt,
                         src_mask=src_causal_mask,
                         tgt_mask=tgt_causal_mask)
        xvc = self.f_c(xvc) ## (N, T, E) x (E, O) -> (N, T, O)
        return xvc
