
from model import FFTSR
import tensorflow as tf
import numpy as np
import cv2
from utils import fft, bicubic, up_sample,imshow,ifft,imshow_spectrum
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = 'images_train/butterfly.bmp'
    img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    img = img/255
    print(img.shape)
    hr_img = fft(img)
    lr_img = fft(up_sample(bicubic(img)))


    # img = img.reshape([1,256,256,1])
    with tf.Session() as sess:
        fftsr = FFTSR(sess, 1e-4, 10000)
        # fftsr.build_model()
        out = fftsr.run(hr_img,lr_img)
        print(type(out))
        out = np.asarray(out)
        out = np.squeeze(out)
        imshow_spectrum(out)

        out = ifft(out)

        out = out *255
        out = np.clip(out)
        print(out.shape)
        imshow(out)
        print(out)