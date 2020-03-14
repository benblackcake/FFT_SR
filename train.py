
from model import FFTSR
import tensorflow as tf
import numpy as np
import cv2
from utils import fft, bicubic, up_sample
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = 'images_train/butterfly.jpg'
    img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    hr_img = fft(img)
    lr_img = fft(up_sample(bicubic(img)))

    magnitude_spectrum = 20 * np.log(np.abs(lr_img))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.show()
    # img = img.reshape([1,256,256,1])
    with tf.Session() as sess:
        fftsr = FFTSR(sess, 1e-4, 10000)
        # fftsr.build_model()
        fftsr.run(hr_img,lr_img)