from typing import Optional, Tuple, Union, Dict, Any
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import  argparse
from .ml_cvnets.cvnets.modules.transformer import TransformerEncoder
from .model_config import get_config

#确保通道数能够被8整除
def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

#在输入上应用二维卷积
class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1, #卷积操作的分组数，默认为 1
        bias: Optional[bool] = False, #是否使用偏置项，默认为 False
        use_norm: Optional[bool] = True, #是否在卷积后使用归一化层（Batch Normalization），默认为 True
        use_act: Optional[bool] = True,  #是否在卷积后使用激活层，默认为 True
    ) -> None:
        super().__init__()

        #将其转换为元组类型
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)

        #使用了卷积核大小的一半作为填充值，以保持输入和输出特征图的大小相同
        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
        )

        #用于按顺序添加不同的卷积层、归一化层和激活层
        block = nn.Sequential()
        #创建卷积层对象conv_layer，并设置其各项参数。然后将其添加到 block 容器中
        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name="conv", module=conv_layer)
        #如果use_norm为True,则创建归一化层对象norm_layer,使用nn.BatchNorm2d创建一个批归一化层,并设置输出通道数为out_channels,动量参数为 0.1。然后将其添加到 block 容器中。
        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
            block.add_module(name="norm", module=norm_layer)
        #如果use_act 为 True，则创建激活层对象 act_layer，使用 nn.SiLU（即 Swish 激活函数）创建一个激活层。然后将其添加到 block 容器中。
        if use_act:
            act_layer = nn.SiLU()
            block.add_module(name="act", module=act_layer)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

#实现了一个反向残差块,通常用于轻量级深度神经网络中
class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float], #扩展比例
        skip_connection: Optional[bool] = True, #是否使用跳跃连接
    ) -> None:
        assert stride in [1, 2]
        #根据扩展比例 expand_ratio 计算隐藏维度 hidden_dim
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        #如果 expand_ratio不等于1，添加一个1x1卷积层，用于将输入通道数扩展到 hidden_dim。
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1
                ),
            )
        #添加一个深度可分离卷积层，其作用是在空间上对每个输入通道进行卷积，然后通过 1x1 卷积层将通道数减少，从而实现轻量级特性。
        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim
            ),
        )

        #添加一个 1x1 卷积层，用于将通道数从 hidden_dim 调整为 out_channels，并且可以选择是否应用归一化层。
        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        #如果使用跳跃连接（use_res_connect=True），则直接返回输入张量加上经过残差块处理后的张量的结果。
        if self.use_res_connect:
            return x + self.block(x)
        #否则，仅返回经过残差块处理后的张量的结果。
        else:
            return self.block(x)


class MobileViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: int = 2,
        head_dim: int = 32,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        patch_h: int = 8,
        patch_w: int = 8,
        conv_ksize: Optional[int] = 3,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        #用于输入通道的卷积
        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )
        #将输入通道数转换为 transformer_dim 维度
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False
        )
        #将 transformer_dim 维度转换回输入通道数
        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1
        )
        #2倍输入通道数的 3x3 卷积层
        conv_3x3_out = ConvLayer(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )

        self.local_rep = nn.Sequential()  #将上述卷积层按顺序添加到其中，用于局部特征处理。
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim
        opts = self.get_opts()

        #创建 global_rep 列表，包含 n_transformer_blocks 个 TransformerEncoder 实例，每个实例表示一个 Transformer 编码器块。
        global_rep = [
            TransformerEncoder(
                opts = opts,
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout
            )
            for _ in range(n_transformer_blocks)
        ]

        # TransformrEncder中有多头注意力 去这里面修改
        global_rep.append(nn.LayerNorm(transformer_dim)) #用于规范化全局表示。
        self.global_rep = nn.Sequential(*global_rep) #将上述列表中的层按顺序组合

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    #设置了模型的一些参数，包括归一化（normalization）的组数（groups）、动量（momentum）、激活函数的名称（name）、是否原地计算（inplace）、负斜率（neg_slope）等。
    def get_opts(self) -> argparse.Namespace:
        opts = argparse.Namespace()
        setattr(opts, "model.normalization.groups", None)
        setattr(opts, "model.normalization.momentum", 0.9)
        setattr(opts, "model.activation.name", "relu")
        setattr(opts, "model.activation.inplace", False)
        setattr(opts, "model.activation.neg_slope", False)
        return opts
    
    #将输入的图像张量 x 转换为一系列图像块（patches），并返回处理后的图像块张量以及一些有关处理过程的信息，如原始图像的尺寸、是否进行了插值处理、总的图像块数量等。
    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h
        batch_size, in_channels, orig_h, orig_w = x.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

    #B 是 batch size，P 是图像块的像素数，N 是总的图像块数量，C 是通道数。
        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [BP, N, C]
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return x, info_dict
    
    #这个方法与 unfolding 方法相反，将处理过的图像块张量 x 转换回原始图像的张量形式。
    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        n_dim = x.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            x.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] -> [B, C, N, P]
        x = x.transpose(1, 3)
        # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
        x = x.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return x

    #它首先对输入的图像张量 x 进行了局部表示（local representation）的处理，并将其转换为图像块。然后通过一系列的全局表示（global representation）层对图像块进行处理，
    #得到更高级的特征表示。最后通过卷积层和融合操作，将原始图像的张量与处理后的图像块的张量进行融合，最终返回一个处理后的特征图像张量。
    def forward(self, x: Tensor) -> Tensor:
        res = x

        fm = self.local_rep(x)

        # 将特征图转化为patches
        patches, info_dict = self.unfolding(fm)
        # print("self.global_rep",self.global_rep)
        # 学习全局表征
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # [B x Patch x Patches x C] -> [B x C x Patches x Patch]
        fm = self.folding(x=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm


class MobileViT(nn.Module):
    def __init__(self, model_cfg: Dict, num_classes: int = 3): #指定模型的配置和分类类别的数量
        super().__init__()

        image_channels = 3
        out_channels = 16

        #将输入图像的通道数从 3 转换为 16，并进行下采样以减小特征图的大小
        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )
        #使用 _make_layer 方法构建多个特征提取层
        self.layer_1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer2"])
        self.layer_3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])
        self.layer_5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer5"])

        #根据配置参数 model_cfg 中的信息创建了一个名为 conv_1x1_exp 的卷积层，用于将特征图的通道数调整为适合后续处理的大小。
        # exp_channels = 768
        self.out_cc = exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels, 960)
        self.conv_1x1_exp = ConvLayer(
            in_channels=out_channels,
            out_channels=exp_channels,
            kernel_size=1
        )
        #创建了一个自适应平均池化层 avgpool，用于将特征图的高度和宽度降到1
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(exp_channels, num_classes)  # 此处 exp_channels 是 conv_1x1_exp 的输出通道数
        )

        self.apply(self.init_parameters)

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> [nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []

        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4)
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim,
            conv_ksize=3
        ))

        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, x: Tensor) -> Tuple[Any, Any]:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)

        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)              #[8,96,7,7]
        x = self.conv_1x1_exp(x)           #[8,384,7,7]
        x = self.avgpool(x)  # 将特征图大小降至 1x1
        x = self.classifier(x)  # 将平均池化后的结果通过分类器

        return x
        # return x_layer2, x
#zcw
    @property
    def layer2_output(self):
        return self._layer2_output

    @layer2_output.setter
    def layer2_output(self, value):
        self._layer2_output = value

def mobile_vit_xx_small(num_classes: int = 3):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt
    config = get_config("xx_small")
    m = MobileViT(config, num_classes=num_classes)
    return m,config


def mobile_vit_x_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt
    config = get_config("x_small")
    m = MobileViT(config, num_classes=num_classes)
    return m,config


def mobile_vit_small(num_classes: int = 3):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt
    config = get_config("small")
    m = MobileViT(config, num_classes=num_classes)
    return m,config


if __name__ == '__main__':
    model,_ = mobile_vit_small(10)
    x = torch.randn(4,3,640,480)
    print(model(x).shape)