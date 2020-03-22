
import numpy as np
import scipy
import cv2
from matplotlib import pyplot as plt


def fft(img):
    npFFT = np.fft.fft2(img)  # Calculate FFT
    npFFTS = np.fft.fftshift(npFFT)  # Shift the FFT to center it

    return npFFTS

def ifft(img):
    npIffts = np.fft.ifftshift(img)
    npIffts =  np.fft.ifft2(npIffts)
    npIffts = np.abs(npIffts)
    return npIffts

def bicubic(img,scale = 2):
    return cv2.resize(img, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)


def up_sample(img,scale =2):
    return cv2.resize(img, None, fx= scale, fy= scale, interpolation=cv2.INTER_CUBIC)


def plt_imshow(img):
    # magnitude_spectrum = 20 * np.log(np.abs(img))
    plt.imshow(img, cmap='gray')
    plt.show()


def imshow(img):
    # magnitude_spectrum = 20 * np.log(np.abs(img))
    cv2.imshow('test',img)
    cv2.waitKey(0)

def imshow_spectrum(fftimg):
    magnitude_spectrum = np.log(np.real(fftimg))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.show()


def L2_loss(y_true,y_pre):
    return np.sum(np.square(y_true-y_pre))
