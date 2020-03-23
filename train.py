
from model import FFTSR
import tensorflow as tf
import numpy as np
import cv2
from utils import fft, bicubic, up_sample,imshow,ifft,imshow_spectrum,plt_imshow
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = 'images_train/butterfly.bmp'
    # img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    print(img.shape)



    # img = img.reshape([1,256,256,1])
    with tf.Session() as sess:
        hr_img = (img)/255.0 *(1e3*1e-5)
        lr_img = (up_sample(bicubic(img)))/255.0 *(1e3*1e-5)
        # lr_img = (up_sample(bicubic(img)))


        # # imshow_spectrum(lr_img)
        fftsr = FFTSR(sess, 1e-4, 15000)

        # fftsr.build_model()
        res = fftsr.run(hr_img[:, :, 0], lr_img[:, :, 0])
        print('res shape: ',res.shape)
        lr_img = lr_img*255/(1e3*1e-5)
        print('lr_img[:,:,0]',lr_img[:,:,0])
        print('lr_img[:,:,0].shape',lr_img[:,:,0].shape)
        lr_img[:,:,0] = lr_img[:,:,0] + res
        print('after add result',lr_img[:,:,0])
        print('lr_img[:,:,0].shape',lr_img[:,:,0].shape)

        print(lr_img.shape)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_YCR_CB2RGB)
        plt_imshow(lr_img)