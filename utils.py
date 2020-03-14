
import numpy as np
import scipy
import cv2

def fft(img):
    npFFT = np.fft.fft2(img)  # Calculate FFT
    npFFTS = np.fft.fftshift(npFFT)  # Shift the FFT to center it

    return npFFTS

def ifft(img):
    npIffts = np.fft.ifftshift(img)
    npIffts =  np.fft.ifft2(npIffts)

    return npIffts

def bicubic(img,scale = 2):
    return  cv2.resize(img, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_NEAREST)


def up_sample(img,scale =2):
    return  cv2.resize(img, None, fx= scale, fy= scale, interpolation=cv2.INTER_NEAREST)
