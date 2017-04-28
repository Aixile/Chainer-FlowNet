import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import cuda, optimizers, serializers, Variable
import math
import six

from chainer.links.caffe.protobuf3 import caffe_pb2 as caffe_pb

def _get_ksize(param):
    if param.kernel_h > 0:
        return param.kernel_h, param.kernel_w
    elif type(param.kernel_size) == int:
        return param.kernel_size
    elif len(param.kernel_size) == 1:
        return param.kernel_size[0]
    else:
        return param.kernel_size


def _get_stride(param):
    if param.stride_h > 0:
        return param.stride_h, param.stride_w
    elif type(param.stride) == int:
        return param.stride
    elif len(param.stride) == 0:
        return 1
    elif len(param.stride) == 1:
        return param.stride[0]
    else:
        return param.stride


def _get_pad(param):
    if param.pad_h > 0:
        return param.pad_h, param.pad_w
    elif type(param.pad) == int:
        return param.pad
    elif len(param.pad) == 0:
        return 0
    elif len(param.pad) == 1:
        return param.pad[0]
    else:
        return param.pad


def _get_num(blob):
    if blob.num > 0:
        return blob.num
    else:
        return blob.shape.dim[0]


def _get_channels(blob):
    if blob.channels > 0:
        return blob.channels
    else:
        return blob.shape.dim[1]


def _get_height(blob):
    if blob.height > 0:
        return blob.height
    elif len(blob.shape.dim) == 2:
        return blob.shape.dim[0]
    elif len(blob.shape.dim) == 4:
        return blob.shape.dim[2]
    else:
        raise RuntimeError(
            '{}-dimentional array is not supported'.format(
                len(blob.shape.dim)))


def _get_width(blob):
    if blob.width > 0:
        return blob.width
    elif len(blob.shape.dim) == 2:
        return blob.shape.dim[1]
    elif len(blob.shape.dim) == 4:
        return blob.shape.dim[3]
    else:
        raise RuntimeError(
            '{}-dimentional array is not supported'.format(
                len(blob.shape.dim)))


class FlowNetS(chainer.Chain):
    def __init__(self):
        super(FlowNetS, self).__init__(
            conv1=L.Convolution2D(6, 64, 7, stride=2, pad=3),
            conv2=L.Convolution2D(64, 128, 5, stride=2, pad=2),
            conv3=L.Convolution2D(128, 256, 5, stride=2, pad=2),
            conv3_1=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv4=L.Convolution2D(256, 512, 3, stride=2, pad=1),
            conv4_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5=L.Convolution2D(512, 512, 3, stride=2, pad=1),
            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv6=L.Convolution2D(512, 1024, 3, stride=2, pad=1),
            conv6_1=L.Convolution2D(1024, 1024, 3, stride=1, pad=1),

            Convolution1=L.Convolution2D(1024, 2, 3, stride=1, pad=1),
            upsample_flow6to5=L.Deconvolution2D(2, 2, 4, stride=2, pad=1),
            deconv5=L.Deconvolution2D(1024, 512, 4, stride=2, pad=1),

            Convolution2=L.Convolution2D(1026, 2, 3, stride=1, pad=1), #512 + 512 + 2
            upsample_flow5to4=L.Deconvolution2D(2, 2, 4, stride=2, pad=1),
            deconv4=L.Deconvolution2D(1026, 256, 4, stride=2, pad=1),

            Convolution3=L.Convolution2D(770, 2, 3, stride=1, pad=1), #512 + 256 + 2
            upsample_flow4to3=L.Deconvolution2D(2, 2, 4, stride=2, pad=1),
            deconv3=L.Deconvolution2D(770, 128, 4, stride=2, pad=1),

            Convolution4=L.Convolution2D(386, 2, 3, stride=1, pad=1), #128 + 256 + 2
            upsample_flow3to2=L.Deconvolution2D(2, 2, 4, stride=2, pad=1),
            deconv2=L.Deconvolution2D(386, 64, 4, stride=2, pad=1),

            Convolution5=L.Convolution2D(194, 2, 3, stride=1, pad=1), #64 + 128 + 2

            #Convolution6=L.Convolution2D(2, 2, 3, stride=1, pad=1)
            )
        self.div_flow = 20

    def get_layer(self, str):
        return getattr(self, str)

    def __call__(self, x):
        out_conv1 = F.leaky_relu(self.conv1(x), slope=0.1)
        out_conv2 = F.leaky_relu(self.conv2(out_conv1), slope=0.1)
        out_conv3 = F.leaky_relu(self.conv3_1(F.leaky_relu(self.conv3(out_conv2), slope=0.1)), slope=0.1)
        out_conv4 = F.leaky_relu(self.conv4_1(F.leaky_relu(self.conv4(out_conv3), slope=0.1)), slope=0.1)
        out_conv5 = F.leaky_relu(self.conv5_1(F.leaky_relu(self.conv5(out_conv4), slope=0.1)), slope=0.1)
        out_conv6 = F.leaky_relu(self.conv6_1(F.leaky_relu(self.conv6(out_conv5), slope=0.1)), slope=0.1)

        flow6 = self.Convolution1(out_conv6)
        flow6_up = self.upsample_flow6to5(flow6)
        out_deconv5 = F.leaky_relu(self.deconv5(out_conv6), slope=0.1)
        concat5 = F.concat([out_conv5, out_deconv5, flow6_up])

        flow5 = self.Convolution2(concat5)
        flow5_up = self.upsample_flow5to4(flow5)
        out_deconv4 = F.leaky_relu(self.deconv4(concat5), slope=0.1)
        concat4 = F.concat([out_conv4, out_deconv4, flow5_up])

        flow4 = self.Convolution3(concat4)
        flow4_up = self.upsample_flow4to3(flow4)
        out_deconv3 = F.leaky_relu(self.deconv3(concat4), slope=0.1)
        concat3 = F.concat([out_conv3, out_deconv3, flow4_up])

        flow3 = self.Convolution4(concat3)
        flow3_up = self.upsample_flow3to2(flow3)
        out_deconv2 = F.leaky_relu(self.deconv2(concat3), slope=0.1)
        concat2 = F.concat([out_conv2, out_deconv2, flow3_up])

        flow2 = self.Convolution5(concat2)
        return flow2 * self.div_flow


    def load_caffe_model(self, model_path):
        net = caffe_pb.NetParameter()
        with open(model_path, 'rb') as model_file:
            net.MergeFromString(model_file.read())
        print("Loaded")
        if net.layer:
            for layer in net.layer:
                if layer.type == 'Convolution':
                    #continue
                    print(layer.name, layer.type)
                    obj =  self.get_layer(layer.name)
                    blobs = layer.blobs
                    param = layer.convolution_param
                    num = _get_num(blobs[0])
                    channels = _get_channels(blobs[0])
                    #print(obj.out_channels)
                    #print(_get_num(blobs[0]))
                    assert obj.ksize == _get_ksize(param)
                    assert obj.stride[0] == _get_stride(param)
                    assert obj.pad[0] == _get_pad(param)
                    assert obj.W.data.shape[1] == _get_channels(blobs[0])
                    assert obj.out_channels == _get_num(blobs[0])

                    n_in = obj.W.data.shape[1]
                    n_out = obj.out_channels
                    part_size = len(blobs[0].data) // param.group

                    obj.W.data[...] = 0
                    for i in six.moves.range(param.group):
                        in_slice = slice(i * n_in // param.group,
                                    (i + 1) * n_in // param.group)
                        out_slice = slice(i * n_out // param.group,
                                    (i + 1) * n_out // param.group)
                        w = obj.W.data[out_slice, in_slice]
                        data = np.array(
                            blobs[0].data[i * part_size:(i + 1) * part_size])
                        w[:] = data.reshape(w.shape)

                    if param.bias_term:
                        obj.b.data[:] = blobs[1].data

                elif layer.type == 'Deconvolution':
                    print(layer.name, layer.type)
                    obj =  self.get_layer(layer.name)
                    blobs = layer.blobs
                    param = layer.convolution_param

                    num = _get_num(blobs[0])
                    channels = _get_channels(blobs[0])

                    assert obj.ksize == _get_ksize(param)
                    assert obj.stride[0] == _get_stride(param)
                    assert obj.pad[0] == _get_pad(param)
                    assert obj.W.data.shape[0] == num
                    assert obj.out_channels == channels # _get_channels(blobs[0])
                    part_size = len(blobs[0].data) #// param.group
                    obj.W.data[...] = 0
                    obj.W.data = np.array(blobs[0].data[0:part_size]).reshape(obj.W.data.shape)

                    if param.bias_term:
                        obj.b.data[:] = blobs[1].data
