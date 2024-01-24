import math
import random
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb
from sklearn.linear_model import Ridge
from torch import nn

from rmp.global_config import DEVICE
from rmp.utils.logger import LoggerMixin


class ModelArch(Enum):
    """This class contains all six investigated models."""

    XGBOOST = "xgboost"
    LSTM = "lstm"
    TRANSFORMER_ENCODER = "transformer"
    LINEAR_OFFLINE = "linear_offline"
    DLINEAR = "dlinear"
    TRANSFORMER_TSF = "tsf-transformer-v2"
    CUSTOM_MODEL = "custom_model"


class YourCustomModel(nn.Module, LoggerMixin):
    """template for adding a new model for respiratory motion prediction."""

    def __init__(self):
        super().__init__()
        self.input_paras = locals()
        # init your model here
        self.scaler = nn.Parameter(torch.tensor(10.0))  # most trivial model
        self.logger.error(
            "Add custom model logic here. Currently, "
            "placeholer 1-parameter model, does not make any sense."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.logger.debug(f"input shape {x.shape=}")  # (batch, seq_len, input_features)
        # do things here
        x = self.scaler * x
        x = x[:, :, -1:]  # placeholder

        self.logger.info(f"{self.scaler=}")
        self.logger.debug(f"final shape {x.shape=}")  # (batch, seq_len, 1)
        self.logger.error(
            "Add forward function logic of your custom model here. Currently, nothing implemented."
        )
        return x


class MovingAvg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """Series decomposition block."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DecompLinear(nn.Module, LoggerMixin):
    """Decomposition Linear Model.

    Originally published in https://arxiv.org/pdf/2205.13504.pdf.
    Code highly inspired by https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py
    """

    def __init__(self, seq_len, pred_len, individual, enc_in):
        super().__init__()
        self.input_paras = locals()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = SeriesDecomp(kernel_size)
        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        inital_x_shape = x.shape
        if x.shape[0] != 1:
            self.logger.debug(f"Input with batch size greater 1 {x.shape=}")
            x = x.reshape(-1, x.shape[-1])
            x = x[:, :, None]
            self.logger.debug(f"Input was reshaped to {x.shape=}")
        else:
            # re shape x
            x = x.permute((1, 2, 0))
        # x: [batch, seq_len, input_features] -> [seq_len, input_features, batch]
        self.logger.debug(f"{x.shape=}")
        if x.shape[-1] != 1:
            raise ValueError(f"{x.shape=}; only implemented for batch size = 1")
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(
            0, 2, 1
        )
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        self.logger.debug(f"{x.shape=}")
        # [ batch * seq_len, 1, pred_horizon] -> [batch, seq_len, pre_horizon]
        if inital_x_shape[0] != 1:
            x = x.reshape((inital_x_shape[0], inital_x_shape[1], self.pred_len))
            return x[:, :, -1:]
        return x.permute((1, 0, 2))[:, :, -1:]  # to [Batch, Output length, Channel]


# lstm-related
class Many2Many(nn.Module, LoggerMixin):
    """Our LSTM implementation. Code and concept were inspired by.

    - Lin et al 2019; DOI 10.1088/1361-6560/ab13fa
    - Lombardo et al 2022; DOI 10.1088/1361-6560/ac60b7, https://github.com/LMUK-RADONC-PHYS-RES/lstm_centroid_prediction;
    """

    def __init__(
        self,
        input_features: int,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_features = input_features
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(
            num_layers=num_layers,
            input_size=input_features,
            hidden_size=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.fcl = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_batch: torch.Tensor, **kwargs):
        # shape of input batch -> # batch, seq_length, input_features
        batch_size, seq_length, input_features = input_batch.size()
        self.logger.debug(f"{batch_size=}, {seq_length=}, {input_features=} ")
        self.h_c = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(DEVICE),
        )
        predictions = torch.zeros((batch_size, seq_length, self.output_dim)).to(DEVICE)
        for step in range(seq_length):
            input_per_step = input_batch[:, step : step + 1, :]
            lstm_out, self.h_c = self.lstm(input_per_step, self.h_c)
            final_hidden_state = lstm_out[:, -1, :].view(
                batch_size, -1
            )  # final in the sense of hidden layer of the last layer in the stacked lstm
            pred_i = self.fcl(final_hidden_state)
            predictions[:, step, :] = pred_i
        self.logger.debug(
            f"{predictions.shape=}"
        )  # shape: batch_size, seq_length, output_features
        return predictions


class LinearOffline(nn.Module, LoggerMixin):
    def __init__(
        self,
        future_steps: int,
        input_features: int,
        output_features: int,
        alpha: float = 1e-05,
        solver: str = "auto",
    ):
        super().__init__()
        self.future_steps = future_steps
        self.input_features = input_features
        self.output_features = output_features
        self.model = Ridge(alpha=alpha, solver=solver, fit_intercept=True)
        self.trained = False
        self.logger.info(f"{self.model=}")

    def forward(self, features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        self.logger.info(f"{features.shape=}, {targets.shape=}")
        self.model.fit(features[:, :], targets[:, -1:])
        self.trained = True
        output = self.model.predict(features[:, :])
        return output[:]

    def predict_(self, features: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise ValueError("model is not trained")
        self.logger.debug(f"{features.shape=}")
        outputs = self.model.predict(features).squeeze()
        self.logger.debug(f"{outputs.shape=}")
        return outputs


class XGBoostTSF(nn.Module, LoggerMixin):
    """XGBOOST for time series forecasting.

    Every element of the sliding input window is considered as a
    separate feature. This concept was proposed by Elsayed et al 2021
    (https://doi.org/10.48550/arXiv.2101.02118).
    """

    def __init__(
        self,
        n_estimators: int,
        max_depth: int,
        subsample_baselearner: float,
        gamma: float,
        min_child_weight: float,
        learning_rate: float,
        reg_lambda: float,
        future_steps: int,
    ):
        super(XGBoostTSF).__init__()
        self.future_steps = future_steps
        self.model = xgb.XGBRegressor(
            learning_rate=learning_rate,
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_lambda=reg_lambda,
            subsample=subsample_baselearner,
            scale_pos_weight=1,
            verbosity=2,
            gpu_id=0,
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            seed=42,
        )
        self.trained = False

    @staticmethod
    def plot_feature_importance(model):
        fig, axis = plt.subplots(1, 1)
        axis.bar(range(len(model.feature_importances_)), model.feature_importances_)
        return fig

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.logger.debug(f"{features.shape=}, {targets.shape=}")
        self.model.fit(features[:, :], targets[:, -1:])
        self.trained = True
        output = self.model.predict(features[:, :])
        return output[:, None]

    def predict_(self, features: torch.Tensor) -> torch.Tensor:
        if not self.trained:
            raise ValueError("Model not trained!")
        outputs = self.model.predict(features)
        return outputs


class PositionalTimestepEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert max_len > 250, f"max_len: {max_len}. Not suitable for long signals"
        self.max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if self.training:
            random_start = random.randint(0, self.max_len - x.size(1))
        else:
            random_start = 0
        x = x + self.pe[:, random_start : random_start + x.size(1)]
        return self.dropout(x)


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer, LoggerMixin):
    """Overwriting pytorch's build-in TransformerEncoderLayer to get attention
    heads."""

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        attention = []
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        if self.norm_first:
            x, attention = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x_, attention = self._sa_block(x, src_mask, src_key_padding_mask)
            x = self.norm1(x + x_)
            x = self.norm2(x + self._ff_block(x))

        return x, attention

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x, attention = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        self.logger.debug(f"{attention.shape=}")
        return self.dropout1(x), attention


class CustomTransformerEncoder(nn.TransformerEncoder, LoggerMixin):
    """Overwriting pytorch's build-in TransformerEncoder to get attention
    heads."""

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        attentions = []
        for mod in self.layers:
            output, attention = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
            attentions.append(attention)
            self.logger.debug(f"{attention.shape=}")
        if self.norm is not None:
            output = self.norm(output)

        return output, attentions


class Transformer(nn.Module, LoggerMixin):
    """Our implementation of an only-encoder Transformer using masked attention
    heads.

    Note that, an initial learning period of 20s is included for each
    signal. Here, no masking is applied as it is excluded from the final
    evaluation.
    """

    def __init__(
        self,
        input_features_dim: int = 25,
        embedding_features: int = 512,
        num_layers: int = 3,
        dropout: float = 0.0,
        n_heads: int = 4,
        future_steps: int = 12,
        pre_training_size: int = 20,
        output_dim: int = 1,
        max_signal_length_s: int = 300,
        scaling_factor: int = 10,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.future_steps = future_steps
        self.input_features_dim = input_features_dim
        self.embedding_features = embedding_features
        self.output_dim = output_dim
        self.pre_training_size = pre_training_size
        self.embedding_layer = nn.Linear(input_features_dim, embedding_features)
        self.encoder_layer = CustomTransformerEncoderLayer(
            d_model=embedding_features, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = CustomTransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_features, output_dim), nn.Tanh()
        )
        self.positional_encoder = PositionalTimestepEncoding(
            d_model=self.embedding_features,
            max_len=max_signal_length_s * 25,  # SPS = 25
        )

    def _generate_square_subsequent_mask(
        self, sz: int, include_training_period: bool
    ) -> torch.tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        if include_training_period:
            assert self.pre_training_size is not None, "Will cause error in mask"
            mask[: self.pre_training_size, : self.pre_training_size] = float(0.0)
        return mask.to(DEVICE)

    def forward(
        self,
        src: torch.Tensor,
        include_training_period: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len, input_features = src.size()
        src = self.embedding_layer(src)
        src = self.positional_encoder(src)
        mask = self._generate_square_subsequent_mask(
            seq_len, include_training_period=include_training_period
        )
        output, _ = self.transformer_encoder(src, mask=mask)
        output = self.decoder(output) * self.scaling_factor
        return output  # note, does also include training phase outputs

    def infer(
        self,
        src: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, src_seq_len, input_features = src.size()
        outputs = []
        src = self.embedding_layer(src)
        src = self.positional_encoder(src)

        for i in range(1, src_seq_len + 1):
            sampled_src = src[:, :i, :]
            mask = self._generate_square_subsequent_mask(
                sampled_src.shape[1], include_training_period=True
            )
            output, attentions = self.transformer_encoder(sampled_src, mask=mask)
            output = output[:, -1:, :]
            output = self.decoder(output) * self.scaling_factor
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs  # note, does not include training phase outputs


class TransformerTSFv2(torch.nn.Module, LoggerMixin):
    """
    Our re-implementation of the Transformer for respiratory motion prediction utilized in Jeong et al 2022.
    (Paper: https://doi.org/10.1371/journal.pone.0275719
     Code: https://github.com/SangWoonJeong/Respiratory-prediction).
    Underlying Transformer architecture is based on https://arxiv.org/abs/2001.08317 and corresponding implementaion of
    Maclean https://github.com/LiamMaclean216/Pytorch-Transfomer

    In contrast to Maclean, we focused on pytorch built-ins (for attention, encoder and decoder).
    Thus, it is more flexible and efficient.
    """

    def __init__(
        self,
        layer_dim_val: int,
        dec_seq_len: int,
        out_seq_len: int,
        n_decoder_layers: int = 1,
        n_encoder_layers: int = 1,
        n_heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dec_seq_len = dec_seq_len
        input_size = 1
        self.output_seq_len = out_seq_len
        if not input_size == 1:
            raise NotImplementedError(
                "This version only supports a batch size equals one"
            )
        # Initiate encoder and Decoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=layer_dim_val,
            nhead=n_heads,
            dim_feedforward=layer_dim_val,
            dropout=dropout,
            batch_first=True,
            activation=F.elu,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=layer_dim_val,
            nhead=n_heads,
            dim_feedforward=layer_dim_val,
            dropout=dropout,
            batch_first=True,
            activation=F.elu,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=n_decoder_layers, norm=None
        )
        self.pos = PositionalEncoding(layer_dim_val)
        # Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(input_size, layer_dim_val)
        self.dec_input_fc = nn.Linear(input_size, layer_dim_val)
        self.out_fc = nn.Linear(dec_seq_len * layer_dim_val, out_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inital_x_shape = x.shape
        if x.shape[0] != 1:
            self.logger.debug(f"Input with batch size greater 1 {x.shape=}")
            x = x.reshape(-1, x.shape[-1])
            x = x[:, :, None]
            self.logger.debug(f"Input was reshaped to {x.shape=}")
        else:
            # re shape x
            x = x.permute((1, 2, 0))
        # x: [batch, seq_len, input_features] -> [seq_len, input_features, batch]
        x = self._operations(x=x)
        if inital_x_shape[0] != 1:
            x = x.reshape((inital_x_shape[0], inital_x_shape[1], self.output_seq_len))
            self.logger.debug(f"input shape {x.size()=}")
            return x
        x = x[None, :, :]
        self.logger.debug(f"final output shape {x.size()=}")
        return x

    def _operations(self, x: torch.Tensor) -> torch.Tensor:
        self.logger.debug(f"input shape {x.size()=}")
        e = self.enc_input_fc(x)
        self.logger.debug(f"input shape {e.size()=}")
        e = self.pos(e)
        self.logger.debug(f"input shape {e.size()=}")
        e = self.encoder(e)
        d = self.dec_input_fc(x[:, -self.dec_seq_len :])
        d = self.decoder(d, memory=e)
        x = self.out_fc(d.flatten(start_dim=1))
        return x


class PositionalEncoding(nn.Module):
    """Taken from
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[: x.size(1), :].squeeze(1)
        return x
