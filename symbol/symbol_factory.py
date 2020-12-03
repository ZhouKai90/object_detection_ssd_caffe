import math
import caffe
from caffe import layers as L
from caffe import params as P
# from caffe.model_libs import *


from .VGG16_half_channels_512 import getSymbol
from .peleenet import PeleeNetBody
from .layer_utils import ConvBNLayer, resBlock

import sys

sys.path.append("..")
from ssd_config import config

vgg16_mbox_source_layers = ['relu4_3', 'relu7', 'multi_feat_1_conv_3x3_relu', 'multi_feat_2_conv_3x3_relu',
                            'multi_feat_3_conv_3x3_relu', 'multi_feat_4_conv_3x3_relu', 'multi_feat_5_conv_3x3_relu']


pelee_raw_source_layers = ['stage3_tb', 'stage4_tb', 'ext1/fe1_2', 'ext1/fe2_2', 'ext1/fe3_2']
# pelee_mbox_source_layers = ['stage4_tb/ext/pm2/res/relu', 'stage4_tb/ext/pm2/res/relu', 'stage4_tb/ext/pm3/res/relu',
#                             'stage4_tb/ext/pm4/res/relu', 'stage4_tb/ext/pm5/res/relu', 'stage4_tb/ext/pm6/res/relu']
pelee_mbox_source_layers = ['stage4_tb/ext/pm1/res/relu', 'stage4_tb/ext/pm2/res/relu', 'stage4_tb/ext/pm3/res/relu',
                            'stage4_tb/ext/pm4/res/relu', 'stage4_tb/ext/pm5/res/relu']


def addSSDHeaderLayer(net, mbox_source_layers):
    # parameters for generating priors.

    # in percent %
    # min_ratio = 20
    # max_ratio = 90
    # step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
    # min_sizes = []
    # max_sizes = []
    # for ratio in range(min_ratio, max_ratio + 1, step):
    #   min_sizes.append(config.minDim * ratio / 100.)
    #   max_sizes.append(config.minDim * (ratio + step) / 100.)
    #
    # min_sizes = [config.minDim * 10 / 100.] + min_sizes     #在min_sizes中加入了一个0.1的比例，这样使得minSize的数量和sourceLaers的数量匹配
    # max_sizes = [config.minDim * 20 / 100.] + max_sizes

    # min_sizes = [32, 64, 128, 192, 256, 512]            #其实base的框的大小完全是个人为的经验值，至于怎么计算得到，都没关系
    min_sizes = [16, 32, 64, 128, 256]            #其实base的框的大小完全是个人为的经验值，至于怎么计算得到，都没关系
    # max_sizes = [64, 128, 192, 256, 512, 640]

    steps = [8, 16, 32, 64, 128, 256, 512]                  #不设置的话priorbox层里面会根据feature map和原始图片的大小去自己计算
    # aspect_ratios = [[1], [1], [1], [1], [1]]
    # aspect_ratios = [[1, 0.2, 0.3, 0.5], [1, 0.2, 0.3, 0.5], [1, 0.2, 0.3, 0.5], [1, 0.2, 0.3, 0.5], [1, 0.2, 0.3, 0.5]]

    aspect_ratios = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    # aspect_ratios = [[2, 3, 1], [2, 3, 1], [2, 3, 1], [2, 3, 1], [2, 3, 1]]
    # aspect_ratios = [[1,0.3], [1,0.3], [1,0.3], [1,0.3], [1,0.3]]
    # L2 normalize .
    normalizations = [-1, -1, -1, -1, -1]                   #决定于那个分支要使用L2 Normalizetion层，一般con4_3要使用Norm层

    # variance used to encode/decode prior bboxes.
    if config.LOSS_PARA.code_type == P.PriorBox.CENTER_SIZE:
      prior_variance = [0.1, 0.1, 0.2, 0.2]                 #variance是对预测box和真实box的误差进行放大，从而增加loss，增大梯度，加快收敛。
    else:
      prior_variance = [0.1]

    flip = True                                            #是否要水平翻转
    # flip = False                                            #是否要水平翻转
    # clip = True                                            #是否要去掉超出边框的部分
    clip = False                                            #是否要去掉超出边框的部分

    mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
            use_batchnorm=config.useBatchnorm, min_sizes=min_sizes,
            # use_batchnorm=config.useBatchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
            aspect_ratios=aspect_ratios, normalizations=normalizations,
            num_classes=config.numClasses, share_location=config.LOSS_PARA.share_location, flip=flip, clip=clip,
            prior_variance=prior_variance, kernel_size=1, pad=0, lr_mult=config.lrMult)
    return mbox_layers

# Add extra layers on top of a "base" network (e.g. VGGNet or Inception).
def addExtraLayersDefault(net, use_batchnorm=True, lr_mult=1, prefix='ext/fe'):
    use_relu = True
    # Add additional convolutional layers.
    from_layer = net.keys()[-1]
    # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
    for i in range(1, 5):
        out_layer = "{}{}_1".format(i)
        ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=256,
                    kernel_size=1, pad=0, stride=1, lr_mult=lr_mult)
        from_layer = out_layer
        pad = 0 if i > 2 else 1
        stride = 1 if i > 2 else 2
        out_layer = "{}{}_2".format(i)
        ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, num_output=512,
                    kernel_size=3, pad=pad, stride=stride, lr_mult=lr_mult)
        from_layer = out_layer
    return net

def addExtraLayersPelee(net, use_batchnorm=True, lr_mult=1, prefix='ext/fe'):
    use_relu = True

    # Add additional convolutional layers.
    from_layer = net.keys()[-1]

    # 5 x 5
    out_layer = '{}/{}1_1'.format(from_layer, prefix)
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = '{}1_2'.format(prefix)
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2,
      lr_mult=lr_mult)

    # 3 x 3
    from_layer = out_layer
    out_layer = '{}2_1'.format(prefix)
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = '{}2_2'.format(prefix)
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

    # 1 x 1
    from_layer = out_layer
    out_layer = '{}3_1'.format(prefix)
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1,
      lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = '{}3_2'.format(prefix)
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 0, 1,
      lr_mult=lr_mult)

def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        use_objectness=False, normalizations=[], use_batchnorm=True, lr_mult=1,
        use_scale=True, min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
        flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0,
        conf_postfix='', loc_postfix='', head_postfix='ext/pm', **bn_param):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}{}_norm".format(head_postfix, i+1)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth:
            if inter_layer_depth[i] > 0:
                inter_name = "{}{}_inter".format(head_postfix, i+1)
                ConvBNLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True, lr_mult=lr_mult,
                      num_output=inter_layer_depth[i], kernel_size=3, pad=1, stride=1, **bn_param)
                from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)

        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create location prediction layer.
        name = "{}{}_mbox_loc{}".format(head_postfix, i+1, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}{}_mbox_conf{}".format(head_postfix, i+1, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes;
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}{}_mbox_priorbox".format(head_postfix, i+1)
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,
                clip=clip, variance=prior_variance, offset=offset)
        if max_size:
            net.update(name, {'max_size': max_size})
        if aspect_ratio:
            net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(name, {'step': step})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(name, {'img_size': img_height})
            else:
                net.update(name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}{}_mbox_objectness".format(head_postfix, i+1)
            num_obj_output = num_priors_per_location * 2;
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = "mbox_objectness"
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers

def getSymbol(net, backbone):
    if backbone == 'peleenet':
        PeleeNetBody(net, from_layer='data')
        addExtraLayersPelee(net, use_batchnorm=False, prefix='ext1/fe')
        #add res prediction layers
        last_base_layer = 'stage4_tb'
        for i, from_layer in enumerate(pelee_raw_source_layers):    #在每个检测分支加入残差模块
            out_layer = '{}/ext/pm{}'.format(last_base_layer, i+1)
            # out_layer = '{}/ext/pm{}'.format(last_base_layer, i+2)
            resBlock(net, from_layer, 256, out_layer, stride=1, use_bn=True)
        symbol = addSSDHeaderLayer(net, pelee_mbox_source_layers)
        return symbol
    else:
        pass

if __name__ == '__main__':
    net = caffe.NetSpec()
    net.data = L.Input(shape=[dict(dim=[1, 3, 224, 224])])
    net = getSymbol(net, 'peleenet')
    with open('./peleenet.prototxt', 'w') as f:
        f.write(str(net.to_proto()))
