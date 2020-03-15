
from model import FFTSR
import tensorflow as tf
import numpy as np
import cv2
from utils import fft, bicubic, up_sample,imshow,ifft
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = 'images_train/butterfly.bmp'
    img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    img = img/255.0
    print(img.shape)
    hr_img = fft(img)*(1e3*1e-5)
    lr_img = fft(up_sample(bicubic(img)))*(1e3*1e-5)


    # img = img.reshape([1,256,256,1])
    with tf.Session() as sess:
        fftsr = FFTSR(sess, 1e-4, 7000)
        # fftsr.build_model()
        out = fftsr.run(hr_img,lr_img)
        print(type(out))
        out = np.asarray(out)
        out = np.squeeze(out)
        
        out = ifft(out)/(1e3*1e-5)
        out = out *255
        print(out.shape)
        imshow(out)
        print(out)