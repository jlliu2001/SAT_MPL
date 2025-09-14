import torch
import torch.nn as nn
from torch.nn import functional as F
from itertools import repeat
import collections.abc
from functools import partial


'''
This record some basic blocks implementation for MAPSeg_MAE
'''


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, aspp_channel, num_classes, ratio, aspp_dilate=[4, 8, 16]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_channel, aspp_dilate),
            nn.ConvTranspose3d(in_channels=aspp_channel, out_channels=int(aspp_channel / 8), kernel_size=ratio,
                               stride=ratio),
            nn.GroupNorm(num_groups=8, num_channels=int(aspp_channel / 8)),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(aspp_channel / 8), int(aspp_channel / 8),
                      3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=int(aspp_channel / 8)),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(aspp_channel / 8), num_classes, 1)
        )
        self._init_weight()

    def forward_org(self, feature):

        return self.classifier(feature)

    def forward(self, feature):
        decoder_map_list=[]
        for i,cla_func in enumerate(self.classifier):
            feature = cla_func(feature)
            if i<len(self.classifier)-1:
                decoder_map_list.append(feature)

        return feature,decoder_map_list

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv3d(in_channels, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv3d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-3:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='trilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, aspp_channel, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = aspp_channel
        modules = []
        modules.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d(5 * out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


def conv3d(in_channels, out_channels, kernel_size, stride_size, bias, padding):
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride_size, bias=bias, padding=padding)


def create_conv(in_channels, out_channels, kernel_size, stride_size, order, num_groups, padding):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input

    Return:
        list of tuple (name, module)
    """
    # print('ocr:',order)
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels,
                           kernel_size, stride_size, bias, padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(
                num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(
                f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride_size=1, order='gcr', num_groups=8, padding=0):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, stride_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(DoubleConv, self).__init__()
        
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups,
                                   padding=padding))

        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups,
                                   padding=padding))


class ExtResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride_size=1, order='cge', num_groups=8, **kwargs):
        super(ExtResNetBlock, self).__init__()

        # first convolution only first convolution have stride
        if stride_size > 1:
            self.conv1 = SingleConv(in_channels, out_channels, kernel_size=4,
                                    stride_size=stride_size, order=order, num_groups=num_groups, padding=0)
        else:
            self.conv1 = SingleConv(in_channels, out_channels, kernel_size=3, stride_size=stride_size,
                                    order=order, num_groups=num_groups, padding=1)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=3,
                                stride_size=1, order=order, num_groups=num_groups, padding=1)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=3, stride_size=1, order=n_order,
                                num_groups=num_groups, padding=1)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out
        #print('conv1.shape:',out.shape)

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)
        #print('non_linearity.shape:',out.shape)

        return out


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, conv_stride_size=1,
                 basic_module=DoubleConv, conv_layer_order='gcr',
                 num_groups=8, padding=0):
        super(Encoder, self).__init__()

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         stride_size=conv_stride_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, x):
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation) followed by a basic module (DoubleConv or ExtResNetBlock).
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upsample (boole): should the input be upsampled
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=(2, 2, 2), basic_module=DoubleConv,
                 conv_layer_order='gcr', num_groups=8, mode='nearest', padding=1, upsample=True):
        super(Decoder, self).__init__()

        if upsample:
            if basic_module == DoubleConv:
                # if DoubleConv is the basic_module use interpolation for upsampling and concatenation joining
                self.upsampling = InterpolateUpsampling(mode=mode)
                # concat joining
                self.joining = partial(self._joining, concat=True)
            else:
                # if basic_module=ExtResNetBlock use transposed convolution upsampling and summation joining
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=conv_kernel_size, scale_factor=scale_factor)
                # sum joining
                self.joining = partial(self._joining, concat=False)
                # adapt the number of in_channels for the ExtResNetBlock
                in_channels = out_channels
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


def create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_stride_size, conv_padding, layer_order, num_groups):
    # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(in_channels, out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_stride_size=conv_stride_size,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding)
        else:
            # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
            encoder = Encoder(f_maps[i - 1], out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups, padding=conv_padding)

        encoders.append(encoder)

    return nn.ModuleList(encoders)


def res_decoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_stride_size, conv_padding, layer_order, num_groups):
    # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(in_channels, out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding)
        else:
            # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
            encoder = Encoder(f_maps[i - 1], out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups, padding=conv_padding)

        encoders.append(encoder)

    return nn.ModuleList(encoders)


def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, upsample):
    # create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv:
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
        # currently strides with a constant stride: (2, 2, 2)

        _upsample = True
        if i == 0:
            # upsampling can be skipped only for the 1st decoder, afterwards it should always be present
            _upsample = upsample

        decoder = Decoder(in_feature_num, out_feature_num,
                          basic_module=basic_module,
                          conv_layer_order=layer_order,
                          conv_kernel_size=conv_kernel_size,
                          num_groups=num_groups,
                          padding=conv_padding,
                          upsample=_upsample)
        decoders.append(decoder)
    return nn.ModuleList(decoders)


class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True

    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, scale_factor=(2, 2, 2)):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor,
                                      padding=1)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x


class Masked_seg(nn.Module):
    """ Masked Autoencoder with ResNet encoder + DeepLab segmentation header
    """

    def __init__(self, cfg):
        super().__init__()

        # --------------------------------------------------------------------------
        # ResNet encoder specifics
        self.cfg = cfg
        to_tuple = _ntuple(self.cfg.model.depth)
        embed_dim = self.cfg.model.embed_dim
        # encoder
        self.local_encoder = create_encoders(in_channels=1, f_maps=to_tuple(embed_dim), basic_module=ExtResNetBlock,
                                             conv_kernel_size=4, conv_stride_size=4, conv_padding=0, layer_order='gcr',
                                             num_groups=32)

        self.CE = nn.CrossEntropyLoss()
        self.seg_decoder = DeepLabHead(in_channels=embed_dim * 2, aspp_channel=embed_dim, num_classes=cfg.train.cls_num,
                                       ratio=4)

    def patchify(self, imgs, p):
        """

        imgs: (N, 1, H, W, D)
        x: (N, H*W*D/P***3, patch_size**3)
        """
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0 and imgs.shape[4] % p == 0
        h, w, d = [i // p for i in self.cfg.data.patch_size]

        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, d, p))
        x = torch.einsum('nchpwqdr->nhwdpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p ** 3))
        return x

    def unpatchify(self, x, p):
        """

        x: (N, H*W*D/P***3, patch_size**3)
        imgs: (N, 1, H, W, D)
        """
        h, w, d = [i // p for i in self.cfg.data.patch_size]

        assert h * w * d == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p))
        x = torch.einsum('nhwdpqr->nhpwqdr', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p, d * p))
        return imgs

    def random_masking(self, x, mask_ratio, p):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        x = self.patchify(x, p)

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask_ = torch.zeros_like(x_masked)
        # generate the binary mask: 0 is keep, 1 is remove

        x_empty = torch.zeros((N, L - len_keep, D)).cuda()
        mask = torch.ones_like(x_empty)
        x_ = torch.cat([x_masked, x_empty], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        mask_ = torch.cat([mask_, mask], dim=1)
        mask_ = torch.gather(
            mask_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x_masked = self.unpatchify(x_, p)

        mask = self.unpatchify(mask_, p)

        return x_masked, mask

    def forward_encoder(self, x, mask_ratio, p):

        # masking: length -> length * mask_ratio
        if mask_ratio > 0:
            x, mask = self.random_masking(x, mask_ratio, p)

        # apply Transformer blocks
        x_list=[]
        for blk in self.local_encoder:
            x = blk(x)
            x_list.append(x)
        if mask_ratio > 0:
            return x, mask,x_list
        else:
            return x,x_list

    def seg_loss(self, label, pred):
        # this version has confidence mask

        loss = DC_and_CE_loss(
            {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
        loss_seg = loss(pred, label)
        return loss_seg

    def cos_regularization(self, pred, tar):
        loss = nn.CosineEmbeddingLoss()

        return loss(pred.flatten(start_dim=2).squeeze(), tar.flatten(start_dim=2).squeeze(),
                    target=torch.ones((pred.shape[1])).cuda())
    # THIS IS THE MAIN FORWARD FUNCTION to generate pseudo label

    def forward(self, coordinates, local_patch, local_label, global_img, global_label, mask_ratio=0,
                pseudo=False, real_label=True):

        # TODO: WARNING: The current implementation only supports batch size of 1
        # the way to extract feature via these coordinates can't be applied to multi batch

        if len(coordinates.shape) == 2 and coordinates.shape[0] == 1:
            coordinates = coordinates[0]
        if not pseudo:
            if real_label:
                # no masked out
                local_latent_1 = self.forward_encoder(
                    local_patch, mask_ratio=0, p=self.cfg.train.local_mae_patch)
                global_latent_1 = self.forward_encoder(global_img, mask_ratio=0,
                                                       p=self.cfg.train.global_mae_patch)
                global_latent_1_zoomed = global_latent_1[:, :, coordinates[0]:coordinates[1],
                                                         coordinates[2]:coordinates[3],
                                                         coordinates[4]:coordinates[5]].clone()
                upsample = nn.Upsample(
                    size=global_latent_1.shape[2:], mode='trilinear', align_corners=True)
                global_latent_1_zoomed = upsample(global_latent_1_zoomed)

                pred_1 = self.seg_decoder(torch.concat(
                    [local_latent_1, global_latent_1_zoomed], dim=1))
                loss_1 = self.seg_loss(local_label, pred_1)
                pred_aux_1 = self.seg_decoder(torch.concat(
                    [global_latent_1, global_latent_1], dim=1))
                loss_aux_1 = self.seg_loss(global_label, pred_aux_1)
                loss_feat_1 = self.cos_regularization(
                    local_latent_1, global_latent_1_zoomed)
                if mask_ratio > 0:
                    # with masked out:
                    local_latent_2, mask_local = self.forward_encoder(local_patch, mask_ratio=mask_ratio,
                                                                      p=self.cfg.train.local_mae_patch)
                    global_latent_2, _ = self.forward_encoder(global_img, mask_ratio=mask_ratio,
                                                              p=self.cfg.train.global_mae_patch)
                    global_latent_2_zoomed = global_latent_2[:, :, coordinates[0]:coordinates[1],
                                                             coordinates[2]:coordinates[3],
                                                             coordinates[4]:coordinates[5]].clone()
                    global_latent_2_zoomed = upsample(global_latent_2_zoomed)
                    pred_2 = self.seg_decoder(torch.concat(
                        [local_latent_2, global_latent_2_zoomed], dim=1))
                    loss_2 = self.seg_loss(local_label, pred_2)

                    pred_aux_2 = self.seg_decoder(torch.concat(
                        [global_latent_2, global_latent_2], dim=1))
                    loss_aux_2 = self.seg_loss(global_label, pred_aux_2)

                    loss_feat_2 = self.cos_regularization(
                        local_latent_2, global_latent_2_zoomed)
                    return loss_1, loss_2, loss_aux_1, loss_aux_2, loss_feat_1, loss_feat_2, pred_1, pred_2, pred_aux_1, mask_local  # , \
                else:
                    return loss_1, loss_aux_1, loss_feat_1, pred_1, pred_aux_1  # , \
            else:
                if mask_ratio > 0:

                    local_latent_2, mask_local = self.forward_encoder(local_patch, mask_ratio=mask_ratio,
                                                                      p=int(self.cfg.train.local_mae_patch))
                    global_latent_1, _ = self.forward_encoder(global_img, mask_ratio=mask_ratio,
                                                              p=int(self.cfg.train.global_mae_patch))

                    global_latent_1_zoomed = global_latent_1[:, :, coordinates[0]:coordinates[1],
                                                             coordinates[2]:coordinates[3],
                                                             coordinates[4]:coordinates[5]].clone()
                    upsample = nn.Upsample(
                        size=global_latent_1.shape[2:], mode='trilinear', align_corners=True)
                    global_latent_1_zoomed = upsample(global_latent_1_zoomed)
                    pred_2 = self.seg_decoder(torch.concat(
                        [local_latent_2, global_latent_1_zoomed], dim=1))

                    loss_2 = self.seg_loss(local_label, pred_2)
                    loss_feat_1 = self.cos_regularization(
                        local_latent_2, global_latent_1_zoomed)
                    pred_aux = self.seg_decoder(torch.concat(
                        [global_latent_1, global_latent_1], dim=1))
                    loss_aux = self.seg_loss(
                        global_label, pred_aux)

                    return loss_2, pred_2, loss_aux, pred_aux, mask_local, loss_feat_1
                else:

                    local_latent_2 = self.forward_encoder(local_patch, mask_ratio=mask_ratio,
                                                          p=int(self.cfg.train.local_mae_patch))
                    global_latent_1 = self.forward_encoder(global_img, mask_ratio=mask_ratio,
                                                           p=int(self.cfg.train.global_mae_patch))

                    global_latent_1_zoomed = global_latent_1[:, :, coordinates[0]:coordinates[1],
                                                             coordinates[2]:coordinates[3],
                                                             coordinates[4]:coordinates[5]].clone()
                    upsample = nn.Upsample(
                        size=global_latent_1.shape[2:], mode='trilinear', align_corners=True)
                    global_latent_1_zoomed = upsample(global_latent_1_zoomed)
                    pred_2 = self.seg_decoder(torch.concat(
                        [local_latent_2, global_latent_1_zoomed], dim=1))

                    loss_2 = self.seg_loss(local_label, pred_2)
                    loss_feat_1 = self.cos_regularization(
                        local_latent_2, global_latent_1_zoomed)
                    pred_aux = self.seg_decoder(torch.concat(
                        [global_latent_1, global_latent_1], dim=1))
                    loss_aux = self.seg_loss(
                        global_label, pred_aux)

                    return loss_2, pred_2, loss_aux, pred_aux, loss_feat_1

        elif pseudo:
            local_latent_1,local_image_embedding_list = self.forward_encoder(
                local_patch, mask_ratio=0, p=self.cfg.train.local_mae_patch)
            global_latent_1,_ = self.forward_encoder(
                global_img, mask_ratio=0, p=self.cfg.train.global_mae_patch)
            global_latent_1_zoomed = global_latent_1[:, :, coordinates[0]:coordinates[1],
                                                     coordinates[2]:coordinates[3],
                                                     coordinates[4]:coordinates[5]].clone()
            upsample = nn.Upsample(
                size=global_latent_1.shape[2:], mode='trilinear', align_corners=True)
            global_latent_1_zoomed = upsample(global_latent_1_zoomed)
            pred_1 = self.seg_decoder(torch.concat(
                [local_latent_1, global_latent_1_zoomed], dim=1))
            # pred_1 = self.seg_decoder(local_latent_1)
            pred_aux = self.seg_decoder(torch.concat(
                [global_latent_1, global_latent_1], dim=1))
            return pred_1, pred_aux,local_image_embedding_list
    def forward_only1(self, coordinates, local_patch, local_label, global_img, global_label, mask_ratio=0,
                      pseudo=False, real_label=True):

        # TODO: WARNING: The current implementation only supports batch size of 1
        # the way to extract feature via these coordinates can't be applied to multi batch

        if len(coordinates.shape) == 2 and coordinates.shape[0] == 1:
            coordinates = coordinates[0]
        if not pseudo:
            if real_label:
                # no masked out
                local_latent_2, mask_local = self.forward_encoder(local_patch, mask_ratio=mask_ratio,
                                                                  p=self.cfg.train.local_mae_patch)
                global_latent_2, _ = self.forward_encoder(global_img, mask_ratio=mask_ratio,
                                                          p=self.cfg.train.global_mae_patch)
                upsample = nn.Upsample(
                    size=global_latent_2.shape[2:], mode='trilinear', align_corners=True)
                global_latent_2_zoomed = global_latent_2[:, :, coordinates[0]:coordinates[1],
                                                         coordinates[2]:coordinates[3],
                                                         coordinates[4]:coordinates[5]].clone()
                global_latent_2_zoomed = upsample(global_latent_2_zoomed)
                pred_2 = self.seg_decoder(torch.concat(
                    [local_latent_2, global_latent_2_zoomed], dim=1))
                loss_2 = self.seg_loss(local_label, pred_2)

                pred_aux_2 = self.seg_decoder(torch.concat(
                    [global_latent_2, global_latent_2], dim=1))
                loss_aux_2 = self.seg_loss(global_label, pred_aux_2)

                loss_feat_2 = self.cos_regularization(
                    local_latent_2, global_latent_2_zoomed)
                return loss_2, loss_aux_2, loss_feat_2, pred_2, pred_aux_2


class EMA_MPL(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.teacher = Masked_seg(cfg=cfg)
        self.student = Masked_seg(cfg=cfg)
        self.cfg = cfg

    def initialize_load(self):
        self.student.load_state_dict(
            torch.load(self.cfg.model.pretrain_model),
            strict=False)
        print('pretrained weights loaded, %s' % self.cfg.model.pretrain_model)

    def _init_ema_weights(self):

        for param in self.teacher.parameters():
            param.detach_()
        mp = list(self.student.parameters())
        mcp = list(self.teacher.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()
        print('EMA weights initialized')

    @torch.no_grad()
    def _update_ema(self, iter):
        if self.cfg.model.large_scale:
            # for the model that was pretrained on large-scale data > 1000
            if iter < self.cfg.train.warmup*100 + 1000:
                # for the first 10 epochs after warmup
                # iteration per epoch is 100 and is never tuned
                # if that was altered, this part should be changed and the performance might be affected
                alpha_teacher = 0.999
            else:
                alpha_teacher = 0.9999
        else:
            # for the model that was pretrained on small-batch data: dozens to hundreds
            if iter < self.cfg.train.warmup*100 + 1000:  # for the first 10 epochs after warmup
                alpha_teacher = 0.99
            # for the 10-30th epochs after warmup
            elif iter >= self.cfg.train.warmup*100 + 1000 and iter < self.cfg.train.warmup*100 + 3000:
                alpha_teacher = 0.999
            else:
                alpha_teacher = 0.9999

        for ema_param, param in zip(self.teacher.parameters(),
                                    self.student.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]
    # TO GET THE PSEUDO LABEL

    @torch.no_grad()
    def get_pseudo_label(self, local_patch, global_img, coordinates):
        pseudo, pseudo_aux = self.teacher(local_patch=local_patch, local_label=None, global_img=global_img,
                                          global_label=None, coordinates=coordinates, pseudo=True)
        return pseudo, pseudo_aux

    @torch.no_grad()
    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        _, pseudo_label = torch.max(ema_softmax, dim=1)

        # Below is a simple way to get the pseudo label with a certain threshold on confidence (prob.)
        # pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        # ps_large_p = pseudo_prob.ge(
        #     0.95).long() == 1
        # pseudo_label *= ps_large_p
        return pseudo_label

    # this is the training loop for source domain

    def train_source(self, cord_src, img_src, label_src, global_src, label_src_aux, src_mask_ratio):
        if src_mask_ratio > 0:
            seg_loss, seg_loss_masked, seg_loss_aux, seg_loss_aux_masked, cos_feat, cos_feat_masked, pred_seg, pred_seg_masked, pred_aux, mask_seg = \
                self.student(cord_src, img_src, label_src, global_src,
                             label_src_aux, src_mask_ratio)

            return seg_loss, seg_loss_masked, seg_loss_aux, seg_loss_aux_masked, cos_feat, cos_feat_masked, pred_seg, pred_seg_masked, pred_aux, mask_seg
        else:
            seg_loss, seg_loss_aux, cos_feat, pred_seg, pred_aux = \
                self.student(cord_src, img_src, label_src, global_src,
                             label_src_aux, src_mask_ratio)
            return seg_loss, seg_loss_aux, cos_feat, pred_seg, pred_aux

    def train_source_only1(self, cord_src, img_src, label_src, global_src, label_src_aux, src_mask_ratio):

        seg_loss, seg_loss_aux, cos_feat, pred_seg, pred_aux = \
            self.student.forward_only1(cord_src, img_src, label_src, global_src,
                                       label_src_aux, src_mask_ratio)
        return seg_loss, seg_loss_aux, cos_feat, pred_seg, pred_aux

    # THIS IS THE TRAINIGN LOOP FOR TARGET DOMAIN

    def train_pseudo(self, cord_tgt, img_tgt, pseudo_label_loc, global_tgt, pseudo_label_global, trg_mask_ratio):
        if trg_mask_ratio > 0:
            pse_seg_loss, pse_seg_pred, pse_seg_loss_aux, pse_seg_pred_aux, pse_seg_mask, pse_cos_feat = \
                self.student(cord_tgt, img_tgt, pseudo_label_loc, global_tgt, pseudo_label_global, trg_mask_ratio,
                             real_label=False)

            return pse_seg_loss, pse_seg_pred, pse_seg_loss_aux, pse_seg_pred_aux, pse_seg_mask, pse_cos_feat
        else:
            pse_seg_loss, pse_seg_pred, pse_seg_loss_aux, pse_seg_pred_aux, pse_cos_feat = \
                self.student(cord_tgt, img_tgt, pseudo_label_loc, global_tgt, pseudo_label_global, trg_mask_ratio,
                             real_label=False)

            return pse_seg_loss, pse_seg_pred, pse_seg_loss_aux, pse_seg_pred_aux, pse_cos_feat

    def forward(self, local_patch, global_img, coordinates):
        pseudo, pseudo_aux = self.student(local_patch=local_patch, local_label=None, global_img=global_img,
                                          global_label=None, coordinates=coordinates, pseudo=True)
        return pseudo, pseudo_aux



class Masked_segV2(nn.Module):
    """ Masked Autoencoder with ResNet encoder + DeepLab segmentation header
    """

    def __init__(self, cfg):
        super().__init__()

        # --------------------------------------------------------------------------
        # ResNet encoder specifics
        self.cfg = cfg
        to_tuple = _ntuple(self.cfg.model.depth)
        embed_dim = self.cfg.model.embed_dim
        # encoder
        self.local_encoder = create_encoders(in_channels=1, f_maps=to_tuple(embed_dim), basic_module=ExtResNetBlock,
                                             conv_kernel_size=4, conv_stride_size=4, conv_padding=0, layer_order='gcr',
                                             num_groups=32)

        self.CE = nn.CrossEntropyLoss()
        self.seg_decoder = DeepLabHead(in_channels=embed_dim * 2, aspp_channel=embed_dim, num_classes=cfg.train.cls_num,
                                       ratio=4)

    def patchify(self, imgs, p):
        """

        imgs: (N, 1, H, W, D)
        x: (N, H*W*D/P***3, patch_size**3)
        """
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0 and imgs.shape[4] % p == 0
        h, w, d = [i // p for i in self.cfg.data.patch_size]

        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, d, p))
        x = torch.einsum('nchpwqdr->nhwdpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p ** 3))
        return x

    def unpatchify(self, x, p):
        """

        x: (N, H*W*D/P***3, patch_size**3)
        imgs: (N, 1, H, W, D)
        """
        h, w, d = [i // p for i in self.cfg.data.patch_size]

        assert h * w * d == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p))
        x = torch.einsum('nhwdpqr->nhpwqdr', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p, d * p))
        return imgs

    def random_masking(self, x, mask_ratio, p):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        x = self.patchify(x, p)

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask_ = torch.zeros_like(x_masked)
        # generate the binary mask: 0 is keep, 1 is remove

        x_empty = torch.zeros((N, L - len_keep, D)).cuda()
        mask = torch.ones_like(x_empty)
        x_ = torch.cat([x_masked, x_empty], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        mask_ = torch.cat([mask_, mask], dim=1)
        mask_ = torch.gather(
            mask_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x_masked = self.unpatchify(x_, p)

        mask = self.unpatchify(mask_, p)

        return x_masked, mask

    def forward_encoder(self, x, mask_ratio, p):

        # masking: length -> length * mask_ratio
        if mask_ratio > 0:
            x, mask = self.random_masking(x, mask_ratio, p)

        # apply Transformer blocks
        x_list=[]
        for blk in self.local_encoder:
            x = blk(x)
            x_list.append(x)
        if mask_ratio > 0:
            return x, mask,x_list
        else:
            return x,None,x_list

    def seg_loss(self, label, pred):
        # this version has confidence mask

        loss = DC_and_CE_loss(
            {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
        loss_seg = loss(pred, label)
        return loss_seg

    def cos_regularization(self, pred, tar):
        loss = nn.CosineEmbeddingLoss()

        return loss(pred.flatten(start_dim=2).squeeze(), tar.flatten(start_dim=2).squeeze(),
                    target=torch.ones((pred.shape[1])).cuda())
    # THIS IS THE MAIN FORWARD FUNCTION to generate pseudo label

    def forward(self, coordinates, local_patch, local_label, global_img, global_label, mask_ratio=0,
                pseudo=False, real_label=True):

        # TODO: WARNING: The current implementation only supports batch size of 1
        # the way to extract feature via these coordinates can't be applied to multi batch

        if len(coordinates.shape) == 2 and coordinates.shape[0] == 1:
            coordinates = coordinates[0]
        # if not pseudo:
        #     if real_label:
        #         # no masked out
        #         local_latent_1 = self.forward_encoder(
        #             local_patch, mask_ratio=0, p=self.cfg.train.local_mae_patch)
        #         global_latent_1 = self.forward_encoder(global_img, mask_ratio=0,
        #                                                p=self.cfg.train.global_mae_patch)
        #         global_latent_1_zoomed = global_latent_1[:, :, coordinates[0]:coordinates[1],
        #                                                  coordinates[2]:coordinates[3],
        #                                                  coordinates[4]:coordinates[5]].clone()
        #         upsample = nn.Upsample(
        #             size=global_latent_1.shape[2:], mode='trilinear', align_corners=True)
        #         global_latent_1_zoomed = upsample(global_latent_1_zoomed)

        #         pred_1 = self.seg_decoder(torch.concat(
        #             [local_latent_1, global_latent_1_zoomed], dim=1))
        #         loss_1 = self.seg_loss(local_label, pred_1)
        #         pred_aux_1 = self.seg_decoder(torch.concat(
        #             [global_latent_1, global_latent_1], dim=1))
        #         loss_aux_1 = self.seg_loss(global_label, pred_aux_1)
        #         loss_feat_1 = self.cos_regularization(
        #             local_latent_1, global_latent_1_zoomed)
        #         if mask_ratio > 0:
        #             # with masked out:
        #             local_latent_2, mask_local = self.forward_encoder(local_patch, mask_ratio=mask_ratio,
        #                                                               p=self.cfg.train.local_mae_patch)
        #             global_latent_2, _ = self.forward_encoder(global_img, mask_ratio=mask_ratio,
        #                                                       p=self.cfg.train.global_mae_patch)
        #             global_latent_2_zoomed = global_latent_2[:, :, coordinates[0]:coordinates[1],
        #                                                      coordinates[2]:coordinates[3],
        #                                                      coordinates[4]:coordinates[5]].clone()
        #             global_latent_2_zoomed = upsample(global_latent_2_zoomed)
        #             pred_2 = self.seg_decoder(torch.concat(
        #                 [local_latent_2, global_latent_2_zoomed], dim=1))
        #             loss_2 = self.seg_loss(local_label, pred_2)

        #             pred_aux_2 = self.seg_decoder(torch.concat(
        #                 [global_latent_2, global_latent_2], dim=1))
        #             loss_aux_2 = self.seg_loss(global_label, pred_aux_2)

        #             loss_feat_2 = self.cos_regularization(
        #                 local_latent_2, global_latent_2_zoomed)
        #             return loss_1, loss_2, loss_aux_1, loss_aux_2, loss_feat_1, loss_feat_2, pred_1, pred_2, pred_aux_1, mask_local  # , \
        #         else:
        #             return loss_1, loss_aux_1, loss_feat_1, pred_1, pred_aux_1  # , \
        #     else:
        #         if mask_ratio > 0:

        #             local_latent_2, mask_local = self.forward_encoder(local_patch, mask_ratio=mask_ratio,
        #                                                               p=int(self.cfg.train.local_mae_patch))
        #             global_latent_1, _ = self.forward_encoder(global_img, mask_ratio=mask_ratio,
        #                                                       p=int(self.cfg.train.global_mae_patch))

        #             global_latent_1_zoomed = global_latent_1[:, :, coordinates[0]:coordinates[1],
        #                                                      coordinates[2]:coordinates[3],
        #                                                      coordinates[4]:coordinates[5]].clone()
        #             upsample = nn.Upsample(
        #                 size=global_latent_1.shape[2:], mode='trilinear', align_corners=True)
        #             global_latent_1_zoomed = upsample(global_latent_1_zoomed)
        #             pred_2 = self.seg_decoder(torch.concat(
        #                 [local_latent_2, global_latent_1_zoomed], dim=1))

        #             loss_2 = self.seg_loss(local_label, pred_2)
        #             loss_feat_1 = self.cos_regularization(
        #                 local_latent_2, global_latent_1_zoomed)
        #             pred_aux = self.seg_decoder(torch.concat(
        #                 [global_latent_1, global_latent_1], dim=1))
        #             loss_aux = self.seg_loss(
        #                 global_label, pred_aux)

        #             return loss_2, pred_2, loss_aux, pred_aux, mask_local, loss_feat_1
        #         else:

        #             local_latent_2 = self.forward_encoder(local_patch, mask_ratio=mask_ratio,
        #                                                   p=int(self.cfg.train.local_mae_patch))
        #             global_latent_1 = self.forward_encoder(global_img, mask_ratio=mask_ratio,
        #                                                    p=int(self.cfg.train.global_mae_patch))

        #             global_latent_1_zoomed = global_latent_1[:, :, coordinates[0]:coordinates[1],
        #                                                      coordinates[2]:coordinates[3],
        #                                                      coordinates[4]:coordinates[5]].clone()
        #             upsample = nn.Upsample(
        #                 size=global_latent_1.shape[2:], mode='trilinear', align_corners=True)
        #             global_latent_1_zoomed = upsample(global_latent_1_zoomed)
        #             pred_2 = self.seg_decoder(torch.concat(
        #                 [local_latent_2, global_latent_1_zoomed], dim=1))

        #             loss_2 = self.seg_loss(local_label, pred_2)
        #             loss_feat_1 = self.cos_regularization(
        #                 local_latent_2, global_latent_1_zoomed)
        #             pred_aux = self.seg_decoder(torch.concat(
        #                 [global_latent_1, global_latent_1], dim=1))
        #             loss_aux = self.seg_loss(
        #                 global_label, pred_aux)

        #             return loss_2, pred_2, loss_aux, pred_aux, loss_feat_1

        if pseudo:
            local_latent_1, _, local_image_embedding_list = self.forward_encoder(
                local_patch, mask_ratio=0, p=self.cfg.train.local_mae_patch)
            global_latent_1, _, _ = self.forward_encoder(
                global_img, mask_ratio=0, p=self.cfg.train.global_mae_patch)
            global_latent_1_zoomed = global_latent_1[:, :, coordinates[0]:coordinates[1],
                                                     coordinates[2]:coordinates[3],
                                                     coordinates[4]:coordinates[5]].clone()
            upsample = nn.Upsample(
                size=global_latent_1.shape[2:], mode='trilinear', align_corners=True)
            global_latent_1_zoomed = upsample(global_latent_1_zoomed)
            pred_1,decoder_map_list = self.seg_decoder(torch.concat(
                [local_latent_1, global_latent_1_zoomed], dim=1))
            # pred_1 = self.seg_decoder(local_latent_1)
            pred_aux,_ = self.seg_decoder(torch.concat(
                [global_latent_1, global_latent_1], dim=1))
            return pred_1, pred_aux,local_image_embedding_list,decoder_map_list
    def forward_only1(self, coordinates, local_patch, local_label, global_img, global_label, mask_ratio=0,
                      pseudo=False, real_label=True):

        # TODO: WARNING: The current implementation only supports batch size of 1
        # the way to extract feature via these coordinates can't be applied to multi batch

        if len(coordinates.shape) == 2 and coordinates.shape[0] == 1:
            coordinates = coordinates[0]
        if not pseudo:
            if real_label:
                # no masked out
                local_latent_2, mask_local = self.forward_encoder(local_patch, mask_ratio=mask_ratio,
                                                                  p=self.cfg.train.local_mae_patch)
                global_latent_2, _ = self.forward_encoder(global_img, mask_ratio=mask_ratio,
                                                          p=self.cfg.train.global_mae_patch)
                upsample = nn.Upsample(
                    size=global_latent_2.shape[2:], mode='trilinear', align_corners=True)
                global_latent_2_zoomed = global_latent_2[:, :, coordinates[0]:coordinates[1],
                                                         coordinates[2]:coordinates[3],
                                                         coordinates[4]:coordinates[5]].clone()
                global_latent_2_zoomed = upsample(global_latent_2_zoomed)
                pred_2 = self.seg_decoder(torch.concat(
                    [local_latent_2, global_latent_2_zoomed], dim=1))
                loss_2 = self.seg_loss(local_label, pred_2)

                pred_aux_2 = self.seg_decoder(torch.concat(
                    [global_latent_2, global_latent_2], dim=1))
                loss_aux_2 = self.seg_loss(global_label, pred_aux_2)

                loss_feat_2 = self.cos_regularization(
                    local_latent_2, global_latent_2_zoomed)
                return loss_2, loss_aux_2, loss_feat_2, pred_2, pred_aux_2



class MPL(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # self.teacher = Masked_seg(cfg=cfg)
        self.student = Masked_segV2(cfg=cfg)
        self.cfg = cfg

    def forward(self, local_patch, global_img, coordinates):
        pseudo, pseudo_aux, local_image_embedding_list,decoder_map_list = self.student(local_patch=local_patch, local_label=None, global_img=global_img,
                                          global_label=None, coordinates=coordinates, pseudo=True)
        return pseudo, pseudo_aux, local_image_embedding_list, decoder_map_list
