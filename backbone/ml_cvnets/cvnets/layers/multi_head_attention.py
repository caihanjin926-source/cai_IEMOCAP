#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import numpy as np
from cvnets.layers.base_layer import BaseLayer
from cvnets.layers.dropout import Dropout
from cvnets.layers.linear_layer import LinearLayer
from utils import logger

class MultiHeadAttention(BaseLayer):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        output_dim: Optional[int] = None,
        coreml_compatible: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        if output_dim is None:
            output_dim = embed_dim
        super().__init__()
        if embed_dim % num_heads != 0:
            logger.error(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )

        self.qkv_proj = LinearLayer(
            in_features=embed_dim, out_features=3 * embed_dim, bias=bias
        )

        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = LinearLayer(
            in_features=embed_dim, out_features=output_dim, bias=bias
        )

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coreml_compatible = coreml_compatible
        self.use_separate_proj_weight = embed_dim != output_dim

    def __repr__(self):
        return "{}(head_dim={}, num_heads={}, attn_dropout={})".format(
            self.__class__.__name__, self.head_dim, self.num_heads, self.attn_dropout.p
        )

    def forward_default(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # [N, S, C]
        b_sz, S_len, in_channels = x_q.shape

        if x_kv is None:
            # self-attention
            # [N, S, C] --> [N, S, 3C] --> [N, S, 3, h, c] where C = hc
            qkv = self.qkv_proj(x_q).reshape(b_sz, S_len, 3, self.num_heads, -1)
            # [N, S, 3, h, c] --> [N, h, 3, S, C]
            qkv = qkv.transpose(1, 3).contiguous()

            # [N, h, 3, S, C] --> [N, h, S, C] x 3
            query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        else:
            T_len = x_kv.shape[1]

            # cross-attention
            # [N, S, C]
            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, S, C] --> [N, S, h, c] --> [N, h, S, c]
            query = (
                query.reshape(b_sz, S_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

            # [N, T, C] --> [N, T, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, T, 2C] --> [N, T, 2, h, c]
            kv = kv.reshape(b_sz, T_len, 2, self.num_heads, self.head_dim)
            # [N, T, 2, h, c] --> [N, h, 2, T, c]
            kv = kv.transpose(1, 3).contiguous()
            key, value = kv[:, :, 0], kv[:, :, 1]

        query = query * self.scaling

        # [N h, T, c] --> [N, h, c, T]
        key = key.transpose(-1, -2)

        # QK^T
        # [N, h, S, c] x [N, h, c, T] --> [N, h, S, T]
        attn = torch.matmul(query, key)

        batch_size, num_heads, num_src_tokens, num_tgt_tokens = attn.shape

        attn_dtype = attn.dtype
        attn_as_float = self.softmax(attn.float())
        attn = attn_as_float.to(attn_dtype)
        attn = self.attn_dropout(attn)
        attn_weight = attn

        # # 绘制第一个头的注意力分数
        # def plot_attention_weights(attn_weights, head_idx=0):
        #     attn_weights = attn_weights[0, head_idx].detach().cpu().numpy()  # 取出第一个样本和指定头的注意力分数
        #     plt.figure(figsize=(10, 8))
        #     sns.heatmap(attn_weights, cmap="viridis")
        #     plt.title(f"Attention Weights - Head {head_idx}")
        #     plt.xlabel("Key Position")
        #     plt.ylabel("Query Position")
        #     plt.show()
        # plot_attention_weights(attn_weight)
        # print("attn_weight",attn_weight.shape)
        # weighted sum
        # [N, h, S, T] x [N, h, T, c] --> [N, h, S, c]
        out = torch.matmul(attn, value)

        # [N, h, S, c] --> [N, S, h, c] --> [N, S, C]
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)
        return out

    def forward(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs
      # ) -> Tuple[Tensor, Tensor]:
    ) -> Tensor:
        if self.coreml_compatible:
            # For CoreML, we follow batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_tracing(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        elif kwargs.get("use_pytorch_mha", False):
            # pytorch uses sequence-first format. Make sure that input is of the form [Sequence, Batch, Hidden dim]
            return self.forward_pytorch(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            # our default implementation format follows batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_default(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

class MultiHeadAttention_withfg(BaseLayer):
    """

    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, S, C_{in})` (32,49,80)
        num_heads (int): Number of heads in multi-head attention
        attn_dropout (Optional[float]): Attention dropout. Default: 0.0
        bias (Optional[bool]): Use bias or not. Default: ``True``

    Shape:
        - Input:
           - Query tensor (x_q) :math:`(N, S, C_{in})` where :math:`N` is batch size, :math:`S` is number of source tokens,
        and :math:`C_{in}` is input embedding dim
           - Optional Key-Value tensor (x_kv) :math:`(N, T, C_{in})` where :math:`T` is number of target tokens
        - Output: same shape as the input

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        output_dim: Optional[int] = None,
        coreml_compatible: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        if output_dim is None:
            output_dim = embed_dim
        super().__init__()
        if embed_dim % num_heads != 0:
            logger.error(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )

        self.qkv_proj = LinearLayer(in_features=embed_dim, out_features=3 * embed_dim, bias=bias)
        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = LinearLayer(in_features=embed_dim, out_features=output_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coreml_compatible = coreml_compatible
        self.use_separate_proj_weight = embed_dim != output_dim

    def __repr__(self):
        return "{}(head_dim={}, num_heads={}, attn_dropout={})".format(
            self.__class__.__name__, self.head_dim, self.num_heads, self.attn_dropout.p
        )

    def forward_default(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # [N, S, C]
        #输入：[32,196,64]  32是batch_size，196是soures tokens,64是input embedding dim
        #x_q:[32,194,64]
        b_sz, S_len, in_channels = x_q.shape


        def f(x):
            return torch.where(x >= 0, x + 1, torch.exp(x))

        def g(k, v, i, j, sigma):
            exponent = -((i - j) ** 2) / (2 * sigma ** 2)
            return torch.matmul(k, v) * np.exp(exponent)

        if x_kv is None:
            # self-attention
            # [N, S, C] --> [N, S, 3C] --> [N, S, 3, h, c] where C = hc
            qkv = self.qkv_proj(x_q).reshape(b_sz, S_len, 3, self.num_heads, -1)
            # [N, S, 3, h, c] --> [N, h, 3, S, C]
            qkv = qkv.transpose(1, 3).contiguous()

            # [N, h, 3, S, C] --> [N, h, S, C] x 3
            query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        else:

            T_len = x_kv.shape[1]

            # cross-attention
            # [N, S, C]
            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, S, C] --> [N, S, h, c] --> [N, h, S, c]
            query = (
                query.reshape(b_sz, S_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

            # [N, T, C] --> [N, T, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, T, 2C] --> [N, T, 2, h, c]
            kv = kv.reshape(b_sz, T_len, 2, self.num_heads, self.head_dim)
            # [N, T, 2, h, c] --> [N, h, 2, T, c]
            kv = kv.transpose(1, 3).contiguous()
            key, value = kv[:, :, 0], kv[:, :, 1]
        #Q、K、V:[32,4,196,16] [N, h, S, c]  其中4是固定的，4*16=64
        query = query * self.scaling
        # [N h, T, c] --> [N, h, c, T]
        key = key.transpose(-1, -2) #[32,4,16,196]

        f_key = f(key) #[32,4,16,196]
        f_value = f(value)#[32,4,196,16]

        c = query.shape[3]
        N=query.shape[0]
        h = query.shape[1]
        s = query.shape[2]
        # 创建 i 和 j 的索引
        i_indices = torch.arange(c).unsqueeze(1)  # [16, 1]
        j_indices = torch.arange(c).unsqueeze(0)  # [1, 16]
        exponent = -((i_indices - j_indices) ** 2) / (2 * 0.5 ** 2)
        # 将 sin_matrix 扩展到与 qk_product 相同的批量和头数
        exponent = exponent.unsqueeze(0).unsqueeze(0)  # [1, 1, 16, 16]
        exponent = exponent.expand(N, h, -1, -1)  # [32, 4, 16, 16]
        Attn= torch.matmul(f_key, f_value) * np.exp(exponent) #[32, 4, 16, 16]

        # QK^T
        # [N, h, S, c] x [N, h, c, T] --> [N, h, S, T]
        # attn = torch.matmul(query, key)
        # # # print("attn.shape",attn.shape)
        batch_size, num_heads, num_src_tokens, num_tgt_tokens = Attn.shape
        if attn_mask is not None:
            # print("attn_mask is not None")
            # attn_mask shape should be the same as attn
            assert list(attn_mask.shape) == [
                batch_size,
                num_src_tokens,
                num_tgt_tokens,
            ], "Shape of attention mask should be [{}, {}, {}]. Got: {}".format(
                batch_size, num_src_tokens, num_tgt_tokens, attn_mask.shape
            )
            # [N, S, T] --> [N, 1, S, T]
            attn_mask = attn_mask.unsqueeze(1)
            attn = Attn + attn_mask

        if key_padding_mask is not None:
            # print("key_padding_mask is not None")
            # Do not attend to padding positions
            # key padding mask size is [N, T]
            assert key_padding_mask.dim() == 2 and list(key_padding_mask.shape) == [
                batch_size,
                num_tgt_tokens,
            ], "Key_padding_mask should be 2-dimension with shape [{}, {}]. Got: {}".format(
                batch_size, num_tgt_tokens, key_padding_mask.shape
            )
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .to(torch.bool),  # [N, T] --> [N, 1, 1, T]
                float("-inf"),
            )

        # attn_dtype = attn.dtype
        # attn_as_float = self.softmax(attn.float())
        # attn = attn_as_float.to(attn_dtype)
        attn = self.attn_dropout(Attn)
        # weighted sum
        # [N, h, S, T] x [N, h, T, c] --> [N, h, S, c]
        out = torch.matmul(query,attn)
        # [N, h, S, c] --> [N, S, h, c] --> [N, S, C]
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)
        return out

    def forward(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs
    ) -> Tensor:
        if self.coreml_compatible:
            # For CoreML, we follow batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_tracing(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        elif kwargs.get("use_pytorch_mha", False):
            # pytorch uses sequence-first format. Make sure that input is of the form [Sequence, Batch, Hidden dim]
            return self.forward_pytorch(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            # our default implementation format follows batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_default(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

class MultiHeadDiffAttention(BaseLayer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        output_dim: Optional[int] = None,
        coreml_compatible: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        if output_dim is None:
            output_dim = embed_dim
        super().__init__()
        if embed_dim % num_heads != 0:
            logger.error(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )
        def lambda_init(depth):
            return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))

        self.qkv_proj = LinearLayer(in_features=embed_dim, out_features=3 * embed_dim, bias=bias)
        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = LinearLayer(in_features=embed_dim, out_features=output_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coreml_compatible = coreml_compatible
        self.use_separate_proj_weight = embed_dim != output_dim
        self.lambda_init = lambda_init(2)
        # Init λ across heads
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)
        self.lambda_v1 = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)
        self.lambda_v2 = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)
    def __repr__(self):
        return "{}(head_dim={}, num_heads={}, attn_dropout={})".format(
            self.__class__.__name__, self.head_dim, self.num_heads, self.attn_dropout.p
        )

    def forward_default(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # [N, S, C]
        #输入：[32,196,64]  32是batch_size，196是soures tokens,64是input embedding dim
        #x_q:[32,196,64]
        b_sz, S_len, in_channels = x_q.shape
        # Compute λ for each head separately
        lambda_1 = torch.exp(torch.sum(self.lambda_k1 * self.lambda_v1, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_2 = torch.exp(torch.sum(self.lambda_k2 * self.lambda_v2, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        def f(x):
            return torch.where(x >= 0, x + 1, torch.exp(x))
        def g(k, v, i, j, sigma):
            exponent = -((i - j) ** 2) / (2 * sigma ** 2)
            return torch.matmul(k, v) * np.exp(exponent)
        if x_kv is None:
            # self-attention
            # [N, S, C] --> [N, S, 3C] --> [N, S, 3, h, c] where C = hc
            qkv = self.qkv_proj(x_q).reshape(b_sz, S_len, 3, self.num_heads, -1)
            # [N, S, 3, h, c] --> [N, h, 3, S, C]
            qkv = qkv.transpose(1, 3).contiguous()

            # [N, h, 3, S, C] --> [N, h, S, C] x 3
            query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        else:

            T_len = x_kv.shape[1]

            # cross-attention
            # [N, S, C]
            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, S, C] --> [N, S, h, c] --> [N, h, S, c]
            query = (
                query.reshape(b_sz, S_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

            # [N, T, C] --> [N, T, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, T, 2C] --> [N, T, 2, h, c]
            kv = kv.reshape(b_sz, T_len, 2, self.num_heads, self.head_dim)
            # [N, T, 2, h, c] --> [N, h, 2, T, c]
            kv = kv.transpose(1, 3).contiguous()
            key, value = kv[:, :, 0], kv[:, :, 1]
        #Q、K、V:[32,4,196,16] [N, h, S, c]  其中4是固定的，4*16=64
        query = query * self.scaling
        key1 = key
        key2 = key
        # [N h, T, c] --> [N, h, c, T]
        key1 = key1.transpose(-1, -2) #[32,4,16,196]
        key2 = key2.transpose(-1, -2)
        value1 = value
        value2 = value
        f_key1 = f(key1) #[32,4,16,196]
        f_value1 = f(value1)#[32,4,196,16]
        f_key2 = f(key2)
        f_value2 = f(value2)

        c = query.shape[3]
        N=query.shape[0]
        h = query.shape[1]
        s = query.shape[2]
        # 创建 i 和 j 的索引
        i_indices = torch.arange(c).unsqueeze(1)  # [16, 1]
        j_indices = torch.arange(c).unsqueeze(0)  # [1, 16]
        exponent = -((i_indices - j_indices) ** 2) / (2 * 0.5 ** 2)
        # 将 sin_matrix 扩展到与 qk_product 相同的批量和头数
        exponent = exponent.unsqueeze(0).unsqueeze(0)  # [1, 1, 16, 16]
        exponent = exponent.expand(N, h, -1, -1)  # [32, 4, 16, 16]
        Attn1= torch.matmul(f_key1, f_value1) * np.exp(exponent) #[32, 4, 16, 16]
        Attn2 = torch.matmul(f_key2, f_value2) * np.exp(exponent)
        Attn = Attn1 - lambda_full * Attn2
        # QK^T
        # [N, h, S, c] x [N, h, c, T] --> [N, h, S, T]
        # attn = torch.matmul(query, key)
        # # # print("attn.shape",attn.shape)
        batch_size, num_heads, num_src_tokens, num_tgt_tokens = Attn.shape
        if attn_mask is not None:
            # print("attn_mask is not None")
            # attn_mask shape should be the same as attn
            assert list(attn_mask.shape) == [
                batch_size,
                num_src_tokens,
                num_tgt_tokens,
            ], "Shape of attention mask should be [{}, {}, {}]. Got: {}".format(
                batch_size, num_src_tokens, num_tgt_tokens, attn_mask.shape
            )
            # [N, S, T] --> [N, 1, S, T]
            attn_mask = attn_mask.unsqueeze(1)
            attn = Attn + attn_mask

        if key_padding_mask is not None:
            # print("key_padding_mask is not None")
            # Do not attend to padding positions
            # key padding mask size is [N, T]
            assert key_padding_mask.dim() == 2 and list(key_padding_mask.shape) == [
                batch_size,
                num_tgt_tokens,
            ], "Key_padding_mask should be 2-dimension with shape [{}, {}]. Got: {}".format(
                batch_size, num_tgt_tokens, key_padding_mask.shape
            )
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .to(torch.bool),  # [N, T] --> [N, 1, 1, T]
                float("-inf"),
            )

        # attn_dtype = attn.dtype
        # attn_as_float = self.softmax(attn.float())
        # attn = attn_as_float.to(attn_dtype)
        attn = self.attn_dropout(Attn)
        # weighted sum
        # [N, h, S, T] x [N, h, T, c] --> [N, h, S, c]
        out = torch.matmul(query,attn)
        # [N, h, S, c] --> [N, S, h, c] --> [N, S, C]
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)
        return out

    def forward(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs
    ) -> Tensor:
        if self.coreml_compatible:
            # For CoreML, we follow batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_tracing(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        elif kwargs.get("use_pytorch_mha", False):
            # pytorch uses sequence-first format. Make sure that input is of the form [Sequence, Batch, Hidden dim]
            return self.forward_pytorch(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            # our default implementation format follows batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_default(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

class MultiHeadAttention_withdiff(BaseLayer):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        output_dim: Optional[int] = None,
        coreml_compatible: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        if output_dim is None:
            output_dim = embed_dim
        super().__init__()
        if embed_dim % num_heads != 0:
            logger.error(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )

        def lambda_init(depth):
            return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))
        self.qkv_proj = LinearLayer(
            in_features=embed_dim, out_features=3 * embed_dim, bias=bias
        )

        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = LinearLayer(
            in_features=embed_dim, out_features=output_dim, bias=bias
        )

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coreml_compatible = coreml_compatible
        self.use_separate_proj_weight = embed_dim != output_dim
        self.lambda_init = lambda_init(2)
        # Init λ across heads
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)
        self.lambda_v1 = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)
        self.lambda_v2 = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)

    def __repr__(self):
        return "{}(head_dim={}, num_heads={}, attn_dropout={})".format(
            self.__class__.__name__, self.head_dim, self.num_heads, self.attn_dropout.p
        )

    def forward_default(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # [N, S, C]
        b_sz, S_len, in_channels = x_q.shape
        # Compute λ for each head separately
        lambda_1 = torch.exp(torch.sum(self.lambda_k1 * self.lambda_v1, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_2 = torch.exp(torch.sum(self.lambda_k2 * self.lambda_v2, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        if x_kv is None:
            # self-attention
            # [N, S, C] --> [N, S, 3C] --> [N, S, 3, h, c] where C = hc
            qkv = self.qkv_proj(x_q).reshape(b_sz, S_len, 3, self.num_heads, -1)
            # [N, S, 3, h, c] --> [N, h, 3, S, C]
            qkv = qkv.transpose(1, 3).contiguous()

            # [N, h, 3, S, C] --> [N, h, S, C] x 3
            query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        else:
            T_len = x_kv.shape[1]

            # cross-attention
            # [N, S, C]
            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, S, C] --> [N, S, h, c] --> [N, h, S, c]
            query = (
                query.reshape(b_sz, S_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

            # [N, T, C] --> [N, T, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, T, 2C] --> [N, T, 2, h, c]
            kv = kv.reshape(b_sz, T_len, 2, self.num_heads, self.head_dim)
            # [N, T, 2, h, c] --> [N, h, 2, T, c]
            kv = kv.transpose(1, 3).contiguous()
            key, value = kv[:, :, 0], kv[:, :, 1]

        query = query * self.scaling
        query1 = query
        query2 = query
        # [N h, T, c] --> [N, h, c, T]
        key = key.transpose(-1, -2)
        key1 = key
        key2 = key

        # QK^T
        # [N, h, S, c] x [N, h, c, T] --> [N, h, S, T]
        attn1 = torch.matmul(query1, key1)
        attn2 = torch.matmul(query2, key2)

        batch_size, num_heads, num_src_tokens, num_tgt_tokens = attn1.shape

        attn_dtype1 = attn1.dtype
        attn_as_float = self.softmax(attn1.float())
        attn1 = attn_as_float.to(attn_dtype1)
        attn_dtype2 = attn2.dtype
        attn_as_float = self.softmax(attn2.float())
        attn2 = attn_as_float.to(attn_dtype2)
        Attn = attn1 - lambda_full * attn2
        attn = self.attn_dropout(Attn)
        # attn_weight = attn

        # # 绘制第一个头的注意力分数
        # def plot_attention_weights(attn_weights, head_idx=0):
        #     attn_weights = attn_weights[0, head_idx].detach().cpu().numpy()  # 取出第一个样本和指定头的注意力分数
        #     plt.figure(figsize=(10, 8))
        #     sns.heatmap(attn_weights, cmap="viridis")
        #     plt.title(f"Attention Weights - Head {head_idx}")
        #     plt.xlabel("Key Position")
        #     plt.ylabel("Query Position")
        #     plt.show()
        # plot_attention_weights(attn_weight)
        # print("attn_weight",attn_weight.shape)
        # weighted sum
        # [N, h, S, T] x [N, h, T, c] --> [N, h, S, c]
        out = torch.matmul(attn, value)

        # [N, h, S, c] --> [N, S, h, c] --> [N, S, C]
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)
        return out

    def forward(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs
      # ) -> Tuple[Tensor, Tensor]:
    ) -> Tensor:
        if self.coreml_compatible:
            # For CoreML, we follow batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_tracing(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        elif kwargs.get("use_pytorch_mha", False):
            # pytorch uses sequence-first format. Make sure that input is of the form [Sequence, Batch, Hidden dim]
            return self.forward_pytorch(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            # our default implementation format follows batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_default(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )