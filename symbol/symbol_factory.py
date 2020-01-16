import math
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.model_libs import *

from .VGG16_half_channels_512 import getSymbol
from .peleenet import PeleeNetBody
from .layer_utils import ConvBNLayer, resBlock
import sys

sys.path.append("..")
from ssd_config import config

vgg16_mbox_source_layers = ['relu4_3', 'relu7', 'multi_feat_1_conv_3x3_relu', 'multi_feat_2_conv_3x3_relu',
                            'multi_feat_3_conv_3x3_relu', 'multi_feat_4_conv_3x3_relu', 'multi_feat_5_conv_3x3_relu']


pelee_raw_source_layers = ['stage3_tb', 'stage4_tb','ext1/fe1_2', 'ext1/fe2_2','ext1/fe3_2']
pelee_mbox_source_layers = ['stage4_tb/ext/pm2/res/relu', 'stage4_tb/ext/pm3/res/relu',
                            'stage4_tb/ext/pm4/res/relu', 'stage4_tb/ext/pm5/res/relu', 'stage4_tb/ext/pm6/res/relu']


def addSSDHeaderLayer(net, mbox_source_layers):
    # parameters for generating priors.

    # in percent %
    min_ratio = 20
    max_ratio = 90
    step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in range(min_ratio, max_ratio + 1, step):
      min_sizes.append(config.minDim * ratio / 100.)

    min_sizes = [config.minDim * 10 / 100.] + min_sizes     #在min_sizes中加入了一个0.1的比例

    steps = [8, 16, 32, 64, 128, 256, 512]                  #不设置的话priorbox层里面会根据feature map和原始图片的大小去自己计算
    # aspect_ratios = [[1], [1], [1], [1], [1]]
    aspect_ratios = [[1, 0.3, 0.5], [1, 0.3, 0.5], [1, 0.3, 0.5], [1, 0.3, 0.5], [1, 0.3, 0.5]]
    # aspect_ratios = [[1,0.3], [1,0.3], [1,0.3], [1,0.3], [1,0.3]]
    # L2 normalize .
    normalizations = [20, -1, -1, -1, -1]                   #决定于那个分支要使用L2 Normalizetion层

    # variance used to encode/decode prior bboxes.
    if config.LOSS_PARA.code_type == P.PriorBox.CENTER_SIZE:
      prior_variance = [0.1, 0.1, 0.2, 0.2]                 #用于加大loss，加快模型的收敛
    else:
      prior_variance = [0.1]

    flip = False                                            #是否要水平翻转
    clip = False                                            #是否要去掉超出边框的部分

    mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
            use_batchnorm=config.useBatchnorm, min_sizes=min_sizes,
            aspect_ratios=aspect_ratios, normalizations=normalizations,
            num_classes=config.numClasses, share_location=config.LOSS_PARA.share_location, flip=flip, clip=clip,
            prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=config.lrMult)
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

def getSymbol(net, backbone):
    if backbone == 'peleenet':
        PeleeNetBody(net, from_layer='data')
        addExtraLayersPelee(net, use_batchnorm=False, prefix='ext1/fe')
        #add res prediction layers
        last_base_layer = 'stage4_tb'
        for i, from_layer in enumerate(pelee_raw_source_layers):    #在每个检测分支加入残差模块
            out_layer = '{}/ext/pm{}'.format(last_base_layer, i+2)
            resBlock(net, from_layer, 256, out_layer, stride=1, use_bn=True)
        symbol = addSSDHeaderLayer(net, pelee_mbox_source_layers)
        return symbol
    else:
        pass

if __name__ == '__main__':
    net = caffe.NetSpec()
    net.data = L.Input(shape=[dict(dim=[1, 3, 224, 224])])
    net = getSymbol(net, 'pelee')
    with open('./peleenet.prototxt', 'w') as f:
        f.write(str(net.to_proto()))
