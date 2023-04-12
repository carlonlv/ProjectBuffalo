"""
This moduel contains the predictor class to model risk.
"""
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn

from ..utility import PositiveFlt, PositiveInt, Prob


class Transformer(nn.Module):
    """ Transformer class to model outlier probability.

    """

    def __init__(self,
                 d_model,
                 output_dimension,
                 softmax_ouput: bool=False,
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
        Initializer for the Transformer class.

        :param d_model: the number of expected features in the encoder/decoder inputs (default=512), embedding size.
        :param output_dimension: the number of expected features in the final inputs, embedding size.
        :param softmax_ouput: Whether to use softmax to output probability (default=False).
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

        :param src: the sequence to the encoder (required). (S, E) for unbatched, (N, S, E) for batched.
        :param tgt: the sequence to the decoder (required). (T, E) for unbatched, (N, T, E) for batched.
        :return: the output of the transformer.
        """
        src_causal_mask = self.model.generate_square_subsequent_mask(src.size(1) if len(src.shape) == 3 else src.size(0)).to(self.device) ## (S, S)
        tgt_causal_mask = self.model.generate_square_subsequent_mask(tgt.size(1) if len(tgt.shape) == 3 else tgt.size(0)).to(self.device) ## (T, T)
        xvc = self.model(src=src,
                         tgt=tgt,
                         src_mask=src_causal_mask,
                         tgt_mask=tgt_causal_mask)
        xvc = self.f_c(xvc) ## (N, T, E) x (E, O) -> (N, T, O)
        return xvc
