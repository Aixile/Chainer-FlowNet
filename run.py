import chainer
from chainer import serializers, Variable, cuda
from flownets import FlowNetS
import cv2
import numpy as np
import argparse

def preprocessing(img):
    img = img.astype('f')
    img = img / 255.0
    img = img.transpose((2, 0, 1))
    return img

def main():
    parser = argparse.ArgumentParser(
        description='Test FlownetS')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument("--load_model", '-m', default='flownets.npz', help='load model')
    args = parser.parse_args()
    if args.gpu >= 0:
        chainer.cuda.get_device(0).use()
    f = FlowNetS()
    serializers.load_npz('flownets.npz', f)

    if args.gpu >=0:
        f.to_gpu()

    img1 = cv2.imread('samples/0000000-imgL.ppm')
    img2 = cv2.imread('samples/0000000-imgR.ppm')
    img1 = cv2.resize(img1, (512,512), cv2.INTER_AREA)
    img2 = cv2.resize(img2, (512,512), cv2.INTER_AREA)
    img1 = preprocessing(img1)
    img2 = preprocessing(img2)
    xin = np.zeros((1, 6, 512, 512), dtype=np.float32)
    xin[0, 0:3, :] = img1
    xin[0, 3:6, :] = img2
    if args.gpu>=0:
        xin = cuda.to_gpu(xin)

    res = f(Variable(xin)).data

    if args.gpu>=0:
        res = cuda.to_cpu(res)
    img=np.zeros((128,128,3))
    img[:,:,0]=res[0, 0] + 128
    img[:,:,2]=res[0, 1] + 128
    img=img.astype(np.uint8)
    cv2.imwrite('samples/out.jpg', img)

if __name__ == '__main__':
    main()
