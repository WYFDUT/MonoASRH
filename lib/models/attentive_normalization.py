import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

import numbers
from timm.models.layers import trunc_normal_

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def caffe2_xavier_init(module, bias=0):
    # `XavierFill` in Caffe2 corresponds to `kaiming_uniform_` in PyTorch
    # Acknowledgment to FAIR's internal code
    kaiming_init(
        module,
        a=1,
        mode='fan_in',
        nonlinearity='leaky_relu',
        distribution='uniform')


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


class HSigmoidv2(nn.Module):
    def __init__(self, inplace: bool = False):
        
        super().__init__()

        self.inplace = inplace

    def forward(self, x: torch.Tensor):
        out = F.relu6((x + 3.), inplace=self.inplace) / 6.
        return out
    

class AttnWeights(nn.Module):
    """ Attention weights for the mixture of affine transformations
        https://arxiv.org/abs/1908.01259
    """
    def __init__(self,
                 attn_mode,
                 num_features,
                 num_affine_trans,
                 num_groups=1,
                 use_rsd=True,
                 use_maxpool=False,
                 eps=1e-3,
                 act_cfg=dict(type="HSigmoidv2")):
        super(AttnWeights, self).__init__()

        # Paper shew that using HardSigmoid activation func will get best res
        # Else we can use Softmax
        # It's superised to see that CA(Coordinate Attention) also use h_sigmoid func
    
        if use_rsd:
            # Paper use rsd method
            use_maxpool = False

        self.num_affine_trans = num_affine_trans
        self.use_rsd = use_rsd
        self.use_maxpool = use_maxpool
        self.eps = eps
        if not self.use_rsd:
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        layers = []
        if attn_mode == 0:
            if act_cfg['type'] == 'HSigmoidv2':
                # nn.Hardsigmoid()
                layers = [nn.Conv2d(num_features, num_affine_trans, 1, bias=False),
                            nn.BatchNorm2d(num_affine_trans),
                            HSigmoidv2(),]
            elif act_cfg['type'] == 'Softmax':
                layers = [nn.Conv2d(num_features, num_affine_trans, 1, bias=False),
                            nn.BatchNorm2d(num_affine_trans),
                            nn.Softmax(dim=1),]
        elif attn_mode == 1:
            if num_groups > 0:
                assert num_groups <= num_affine_trans
                if act_cfg['type'] == 'HSigmoidv2':
                    # nn.Hardsigmoid()
                    layers = [nn.Conv2d(num_features, num_affine_trans, 1, bias=False),
                                nn.GroupNorm(num_channels=num_affine_trans,
                                                num_groups=num_groups),
                                HSigmoidv2(),]
                elif act_cfg['type'] == 'Softmax':
                    layers = [nn.Conv2d(num_features, num_affine_trans, 1, bias=False),
                                nn.GroupNorm(num_channels=num_affine_trans,
                                                num_groups=num_groups),
                                nn.Softmax(dim=1),]
            else:
                if act_cfg['type'] == 'HSigmoidv2':
                    # nn.Hardsigmoid()
                    layers = [nn.Conv2d(num_features, num_affine_trans, 1, bias=False),
                                nn.BatchNorm2d(num_affine_trans),
                                HSigmoidv2(),]
                elif act_cfg['type'] == 'Softmax':
                    layers = [nn.Conv2d(num_features, num_affine_trans, 1, bias=False),
                                nn.BatchNorm2d(num_affine_trans),
                                nn.Softmax(dim=1),]

        else:
            raise NotImplementedError("Unknow attention weight type")

        self.attention = nn.Sequential(*layers)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        if self.use_rsd:
            var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True)
            y = mean * (var + self.eps).rsqrt()

            # var = torch.var(x, dim=(2, 3), keepdim=True)
            # y *= (var + self.eps).rsqrt()
        else:
            y = self.avgpool(x)
            if self.use_maxpool:
                y += F.max_pool2d(x, (h, w), stride=(h, w)).view(b, c, 1, 1)
        return self.attention(y).view(b, self.num_affine_trans)


class AttnBatchNorm2d(nn.BatchNorm2d):
    """ Attentive Normalization with BatchNorm2d backbone
        https://arxiv.org/abs/1908.01259
    """
    # Usually we chooses num_affine_trans as num_features // ratio, ratio uses 16 normally

    # _abbr_ = "AttnBN2d"

    def __init__(self,
                 num_features,
                 num_affine_trans,
                 attn_mode=0,
                 eps=1e-5,
                 momentum=0.1,
                 track_running_stats=True,
                 use_rsd=True,
                 use_maxpool=False,
                 eps_var=1e-3,
                 act_cfg=dict(type="HSigmoidv2")):
        super(AttnBatchNorm2d, self).__init__(num_features,
                                              affine=False,
                                              eps=eps,
                                              momentum=momentum,
                                              track_running_stats=track_running_stats)

        self.num_affine_trans = num_affine_trans
        self.attn_mode = attn_mode
        self.use_rsd = use_rsd
        self.eps_var = eps_var
        self.act_cfg = act_cfg

        self.weight_ = nn.Parameter(
            torch.Tensor(num_affine_trans, num_features))
        self.bias_ = nn.Parameter(
            torch.Tensor(num_affine_trans, num_features))

        self.attn_weights = AttnWeights(attn_mode,
                                        num_features,
                                        num_affine_trans,
                                        use_rsd=use_rsd,
                                        use_maxpool=use_maxpool,
                                        eps=eps_var,
                                        act_cfg=act_cfg)

        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.weight_, 1., 0.1)
        nn.init.normal_(self.bias_, 0., 0.1)

    def forward(self, x):
        output = super(AttnBatchNorm2d, self).forward(x)
        size = output.size()
        y = self.attn_weights(x)  # b x k

        weight = y @ self.weight_  # b x c
        bias = y @ self.bias_  # b x c
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias


class AttnGroupNorm(nn.Module):
    """Attentive Normalization with GroupNorm backbone
        https://arxiv.org/abs/1908.01259
    """
    # __constants__ = ['num_groups', 'num_features', 'num_affine_trans', 'eps',
    #                 'weight', 'bias']
    # _abbr_ = "AttnGN"

    def __init__(self,
                 num_features,
                 num_affine_trans,
                 num_groups,
                 num_groups_attn=1,
                 attn_mode=1,
                 eps=1e-5,
                 use_rsd=True,
                 use_maxpool=False,
                 eps_var=1e-3,
                 act_cfg=dict(type="HSigmoidv2")):
        super(AttnGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        self.num_affine_trans = num_affine_trans
        self.eps = eps
        self.affine = True
        self.weight_ = nn.Parameter(
            torch.Tensor(num_affine_trans, num_features))
        self.bias_ = nn.Parameter(
            torch.Tensor(num_affine_trans, num_features))

        self.register_parameter('weight', None)
        self.register_parameter('bias', None)

        self.attention_weights = AttnWeights(attn_mode,
                                        num_features,
                                        num_affine_trans,
                                        num_groups=num_groups_attn,
                                        use_rsd=use_rsd,
                                        use_maxpool=use_maxpool,
                                        eps=eps_var,
                                        act_cfg=act_cfg)
        self.group_norm = nn.GroupNorm(self.num_groups, self.num_features, self.eps)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight_, 1., 0.1)
        nn.init.normal_(self.bias_, 0., 0.1)

    def forward(self, x):
        #output = F.group_norm(
            #x, self.num_groups, self.weight, self.bias, self.eps)
        # This code seems to be a little bit strange, maybe it will lead to grad exploting
        output = self.group_norm(x)
        size = output.size()

        y = self.attention_weights(x)

        weight = y @ self.weight_
        bias = y @ self.bias_

        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias

    def extra_repr(self):
        return '{num_groups}, {num_features}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)
    

class AttnLayerNorm(nn.LayerNorm):
    """ Attentive Normalization with LayerNorm backbone
        https://arxiv.org/abs/1908.01259
    """
    # normalized_shape is input shape from an expected input of size
    # the input size should be [∗×normalized_shape[0]×normalized_shape[1]×…×normalized_shape[−1]]
    # for detail plz visit https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm

    # _abbr_ = "AttnLN"

    def __init__(self,
                 normalized_shape,
                 num_affine_trans,
                 eps=1e-5,
                 device=None,
                 dtype=None):
        assert isinstance(
            normalized_shape, numbers.Integral
        ), f'only integral normalized shape supported {normalized_shape}'
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=False,
            device=device,
            dtype=dtype)

        affine_shape = tuple([num_affine_trans] + list(self.normalized_shape))

        self.weight_ = nn.Parameter(torch.empty(affine_shape, **factory_kwargs))
        self.bias_ = nn.Parameter(torch.empty(affine_shape, **factory_kwargs))

        self.attn_weights = nn.Sequential(
            nn.Linear(normalized_shape, num_affine_trans),
            nn.Softmax(dim=-1)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.weight_, 1., 0.1)
        nn.init.normal_(self.bias_, 0., 0.1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: B N C
        assert x.ndim == 3

        output = super().forward(x)

        y = self.attn_weights(x)  # B N k

        weight = y @ self.weight_  # B N C
        bias = y @ self.bias_  # B N C

        return weight * output + bias
    

if __name__ == "__main__":
    test = torch.rand([4, 256, 256, 256]).to("cuda:0")
    model = AttnGroupNorm(num_features=256, num_affine_trans=16, num_groups=8).to("cuda:0")
    # model = AttnBatchNorm2d(num_features=256, num_affine_trans=16).to("cuda:0")
    output = model(test)
    print(output.shape, output[0, 0, ...])