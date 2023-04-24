import tensorflow as tf
from san_model import utils

def soca(input_tens, channel, reduction=8):
    batchSize, h, w, c = input_tens.shape
    N = int(h * w)
    min_h = min(h, w)
    h1 = 1000
    w1 = 1000
    if h < h1 and w < w1:
        x_sub = input_tens
    elif h < h1 and w > w1:
        W = (w - w1) // 2
        x_sub = input_tens[:,:,W:(W + w1),:]
    elif w<w1 and h>h1:
        H = (h-h1) // 2
        x_sub = input_tens[:,H:(H+h1),:,:]
    else:
        H = (h-h1) // 2
        W = (w-w1) // 2
        x_sub = input_tens[:,H:(H+h1),W:(W+w1),:]

    cov_mat = utils.covpool(x_sub)
    cov_mat_sqrt = utils.sqrtm(cov_mat, iterN=5)
    cov_mat_sum = tf.math.reduce_mean(cov_mat_sqrt, axis=1).reshape(batchSize,c,1,1)
    y_conv = utils.convdu(input_tens=cov_mat_sum, channel=channel,reduction=reduction)
    return y_conv*input_tens



def _NonLocalBlockND(in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
    assert dimension in [1, 2, 3]
    assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

    if inter_channels is None:
        inter_channels = in_channels // 2
        if inter_channels == 0:
            inter_channels == 1
            
    if dimension == 3:




