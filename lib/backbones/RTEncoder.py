'''
by wyf
'''

import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from .SCNet_arch import Shift8


__all__ = ['HybridEncoder']


def get_activation(act: str, inpace: bool=True):
    '''get activation'''
    act = act.lower()
    if act == 'silu':
        m = nn.SiLU()
    elif act == 'mish':
        m = nn.Mish()
    elif act == 'relu':
        m = nn.ReLU()
    elif act == 'leaky_relu':
        m = nn.LeakyReLU()
    elif act == 'silu':
        m = nn.SiLU()
    elif act == 'gelu':
        m = nn.GELU()
    elif act is None:
        m = nn.Identity()
    elif isinstance(act, nn.Module):
        m = act
    else:
        raise RuntimeError('')  
    if hasattr(m, 'inplace'):
        m.inplace = inpace
    return m 
    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DilatedConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, dilation=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            dilation=padding if dilation is None else dilation,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    

class ResidualBlockShiftv2(nn.Module):
    def __init__(self, num_feat, res_scale=1.0, act=None) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=1, bias=False)
        self.conv3 = ConvNormLayer(num_feat, num_feat, 1, 1, padding=0, act=None)
        self.act1 = nn.Identity() if act is None else get_activation(act) 
        self.act2 = nn.Identity() if act is None else get_activation(act)
        self.shift = Shift8(groups=num_feat//8, stride=1)
        self.norm = nn.BatchNorm2d(num_feat)    #AttnBatchNorm2d(num_feat, 10, momentum=0.03, eps=0.001)
        
    def forward(self, x):
        identity = self.conv3(x)
        out = self.norm(self.conv2(self.act1(self.shift(self.conv1(x)))))
        out = identity + out * self.res_scale
        return self.act2(out)


class IncConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=7, act='relu'):
        super().__init__()
        assert ch_in == ch_out
        self.ch_in = ch_in
        self.ch_out = ch_out
        
        self.pointwise1 = nn.Conv2d(ch_in, ch_out, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2))
        self.norm1 = nn.BatchNorm2d(ch_out)
        self.act1 = nn.Identity() if act is None else get_activation(act) 
        self.depthwise1 = nn.Conv2d(ch_in, ch_out, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))
        self.norm2 = nn.BatchNorm2d(ch_out)
        self.act2 = nn.Identity() if act is None else get_activation(act) 
        
        self.pointwise2 = nn.Conv2d(ch_in, ch_out, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2))
        self.norm3 = nn.BatchNorm2d(ch_out)
        self.act3 = nn.Identity() if act is None else get_activation(act) 
        self.depthwise2 = nn.Conv2d(ch_in, ch_out, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))
        self.norm4 = nn.BatchNorm2d(ch_out)
        self.act4 = nn.Identity() if act is None else get_activation(act) 
        
    def forward(self, x):
        x = self.act1(self.norm1(self.pointwise1(x)))
        x = self.act2(self.norm2(self.depthwise1(x)))
        x = self.act3(self.norm3(self.pointwise2(x)))
        x = self.act4(self.norm4(self.depthwise2(x)))
        return x
        

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', use_se=False):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 
        if use_se:
            self.post_se = SELayer(ch_out, reduction=4)
        else:
            self.post_se = None

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        if self.post_se is not None:
            return self.post_se(self.act(y))
        else:
            return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act, use_se=True) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='mish',
                 eval_spatial_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )
        self.output_proj = nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=1, bias=False),
                        nn.BatchNorm2d(hidden_dim//2),
                        nn.Identity() if act is None else get_activation(act),
                        ResidualBlockShiftv2(hidden_dim//2, 1.0, act)
                    )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(IncConvBlock(hidden_dim, hidden_dim, 7, act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        '''channel mapping'''
        
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='bilinear')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)
        outs = [inner_outs[0]]
        return self.output_proj(outs[0])


if __name__ == "__main__":
    encoder = HybridEncoder(in_channels=[64, 128, 256, 512],
                                    feat_strides=[4, 8, 16, 32],
                                    hidden_dim=128,
                                    nhead=8,
                                    dim_feedforward=256,
                                    dropout=0.0,
                                    enc_act='gelu',
                                    use_encoder_idx=[3],
                                    num_encoder_layers=1,
                                    pe_temperature=10000,
                                    expansion=0.5,
                                    depth_mult=1.0,
                                    act='mish',
                                    eval_spatial_size=[96*4, 320*4])
    test_input = []
    '''
    encoder.eval()
    for m in encoder.modules():
        if hasattr(m, 'convert_to_deploy'):
            m.convert_to_deploy() 
    '''
    for i in range(4):
        test_input.append(torch.randn([2, 64*(2**i), 96//(2**i), 320//(2**i)])) 
    res = encoder(test_input)
    print(res.shape)
    total_params = sum(p.numel() for p in encoder.parameters())
    total_params += sum(p.numel() for p in encoder.buffers())
    print(f'{total_params:,} total parameters.')
    print(f'{total_params/(1024*1024):.2f}M total parameters.')
    total_trainable_params = sum(
        p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(f'{total_trainable_params/(1024*1024):.5f}M training parameters.')
    # print(encoder)
        # print(len(res))
    # breakpoint()

    test_input = [torch.randn(1, 64*(2**i), 96//(2**i), 320//(2**i)) for i in range(4)]
    
    # Use torch.profiler to profile the model
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
        with_flops=True
    ) as prof:
        encoder(test_input)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

    pass