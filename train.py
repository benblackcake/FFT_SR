
from model import FFTSR
import tensorflow as tf
import numpy as np
import cv2
from utils import fft, bicubic, up_sample,imshow,ifft,imshow_spectrum,plt_imshow
import argparse
from tqdm import tqdm,trange
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for Adam.')
    parser.add_argument('--epoch', type=int, default='1000000', help='How many iterations ')
    args = parser.parse_args()


if __name__ == '__main__':
    img = 'images_train/butterfly.bmp'
    # img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    print('origin_img_shape',img.shape)



    # img = img.reshape([1,256,256,1])
    with tf.Session() as sess:
        hr_img = (img)/255.0 *(1e3*1e-5)
        lr_img = (up_sample(bicubic(img)))/255.0 *(1e3*1e-5)
        # lr_img = (up_sample(bicubic(img)))


        # # imshow_spectrum(lr_img)
        fftsr = FFTSR(sess, 1e-4, 15000)

        # fftsr.build_model()
        res = fftsr.run(hr_img[:, :, 0], lr_img[:, :, 0])
        sr_img = lr_img
        print('res shape: ',res.shape)
        sr_img = sr_img*255/(1e3*1e-5)
        lr_img = lr_img*255/(1e3*1e-5)
        print('sr_img[:,:,0]',sr_img[:,:,0])
        print('sr_img[:,:,0].shape',sr_img[:,:,0].shape)
        sr_img[:,:,0] = sr_img[:,:,0] + res
        print('after add result',sr_img[:,:,0])
        print('sr_img[:,:,0].shape',sr_img[:,:,0].shape)

        print(lr_img.shape)
        sr_img = sr_img.astype(np.uint8)
        lr_img = lr_img.astype(np.uint8)
        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_YCR_CB2RGB)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_YCR_CB2RGB)

        cv2.imwrite('sr_.jpg',sr_img)
        cv2.imwrite('lr_.jpg',lr_img)

        plt_imshow(sr_img)
        plt_imshow(lr_img)