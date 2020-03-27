
from model import FFTSR
import tensorflow as tf
import numpy as np
import cv2
from utils import fft, bicubic, up_sample,imshow,ifft,imshow_spectrum,plt_imshow
from utils_crop_sub_image import*
import argparse
from tqdm import tqdm,trange
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for Adam.')
    parser.add_argument('--epoch', type=int, default='10000', help='How many iterations ')
    parser.add_argument('--image-size', type=int, default=33, help='Size of random crops used for training samples.')
    parser.add_argument('--c-dim', type=int, default=3, help='The size of channel')
    parser.add_argument('--scale', type=int, default=2, help='the size of scale factor for preprocessing input image')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint', help='Name of checkpoint directory')
    parser.add_argument('--result_dir', type=str, default='result', help='Name of result directory')
    parser.add_argument('--test-img', type=str, default='', help='test_img')
    parser.add_argument('--is-train', action='store_true', help='training')
    parser.add_argument('--batch-size', type=int, default=128, help='Mini-batch size.')


    args = parser.parse_args()

    lr_images = tf.placeholder(tf.float32, [None,33,33], name='lr_images')
    hr_images = tf.placeholder(tf.float32, [None,33,33], name='hr_images')

    fftsr = FFTSR(learning_rate=args.learning_rate)
    sr_forward = fftsr.model(lr_images)
    # sr_forward = fftsr.model(lr_images)
    loss = fftsr.loss_function(hr_images - lr_images, sr_forward)
    sr_opt = fftsr.optimizer(loss)



    with tf.Session() as sess:

        nx, ny = input_setup(args.image_size, args.scale, True, args.checkpoint_dir)
        data_dir = checkpoint_dir(True, args.checkpoint_dir)
        input_, label_ = read_data(data_dir)
        print('input_.shape',input_.shape)
        print('input_.shape',label_.shape)
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())
        # init = (tf.global_variables_initializer())
        # sess.run(init,feed_dict={lr_images})
        for epoch in tqdm(range(args.epoch)):

            batch_idxs = len(input_) // args.batch_size
            # print(len(input_))
            for idx in range(0, batch_idxs):
                batch_images = input_[idx * args.batch_size : (idx + 1) * args.batch_size]
                batch_labels = label_[idx * args.batch_size : (idx + 1) * args.batch_size]

                b_images = np.reshape(batch_images[:,:,:,0],[128,33,33])
                b_labels = np.reshape(batch_labels[:,:,:,0],[128,33,33])
                # print(b_images.shape)
                # print(b_labels.shape)

                _, err = sess.run([sr_opt, loss],
                                feed_dict={lr_images: b_images, hr_images: b_labels})
                print('error: ',err)


if __name__ == '__main__':

    main()
    # img = 'images_train/butterfly.bmp'
    # # img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # print('origin_img_shape',img.shape)
    #
    #
    #
    # # img = img.reshape([1,256,256,1])
    # with tf.Session() as sess:
    #     hr_img = (img)/255.0 *(1e3*1e-5)
    #     lr_img = (up_sample(bicubic(img)))/255.0 *(1e3*1e-5)
    #     # lr_img = (up_sample(bicubic(img)))
    #
    #
    #     # # imshow_spectrum(lr_img)
    #     fftsr = FFTSR(sess, 1e-4, 15000)
    #
    #     # fftsr.build_model()
    #     res = fftsr.run(hr_img[:, :, 0], lr_img[:, :, 0])
    #     sr_img = lr_img
    #     print('res shape: ',res.shape)
    #     sr_img = sr_img*255/(1e3*1e-5)
    #     lr_img = lr_img*255/(1e3*1e-5)
    #     print('sr_img[:,:,0]',sr_img[:,:,0])
    #     print('sr_img[:,:,0].shape',sr_img[:,:,0].shape)
    #     sr_img[:,:,0] = sr_img[:,:,0] + res
    #     print('after add result',sr_img[:,:,0])
    #     print('sr_img[:,:,0].shape',sr_img[:,:,0].shape)
    #
    #     print(lr_img.shape)
    #     sr_img = sr_img.astype(np.uint8)
    #     lr_img = lr_img.astype(np.uint8)
    #     sr_img = cv2.cvtColor(sr_img, cv2.COLOR_YCR_CB2RGB)
    #     lr_img = cv2.cvtColor(lr_img, cv2.COLOR_YCR_CB2RGB)
    #
    #     plt_imshow(sr_img)
    #     plt_imshow(lr_img)