
from model import FFTSR
import tensorflow as tf
import numpy as np
import cv2
from utils import fft, bicubic, up_sample,imshow,ifft,imshow_spectrum
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = 'images_train/butterfly.bmp'
    img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)

    print(img.shape)



    # img = img.reshape([1,256,256,1])
    with tf.Session() as sess:
        hr_img = (img)/255.0 *(1e3*1e-5)
        lr_img = (up_sample(bicubic(img)))/255.0 *(1e3*1e-5)
        # imshow_spectrum(lr_img)
        fftsr = FFTSR(sess, 1e-4, 100000)

        # fftsr.build_model()
        fftsr.run(hr_img,lr_img)

        # out = fftsr.pred
        # print(out)