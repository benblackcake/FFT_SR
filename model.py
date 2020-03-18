

import tensorflow as tf
from utils import fft,L2_loss
import numpy as np
from utils import imshow,imshow_spectrum

class FFTSR:

    def __init__(self, sess, learning_rate, epoch):
        self.sess =sess
        self.epoch = epoch
        # self.images = images
        self.learning_rate = learning_rate
        self.build_model()


    def build_model(self):
        self.images = tf.placeholder(tf.float32, [256, 256], name='input_img')
        self.label = tf.placeholder(tf.float32, [256, 256], name='HR_img')

        self.image_matrix = tf.reshape(self.images, shape=[-1, 256, 256, 1])

        self.pred = self.model()
        self.loss = tf.nn.l2_loss(self.label - self.pred)
        # squared_deltas = tf.square(self.label - self.pred)
        # self.loss = L2_loss(self.label, self.pred)
        # print(self.pred)
        # self.loss = tf.reduce_mean(tf.square(self.label - self.pred))
        # print('build_model_image_shape',self.images)


    def model(self):
        # x = None
        f1,self.spatial_c1,self.spectral_c1 = self.fft_conv_pure(self.image_matrix,filters=5,width=256,height=256)
        # f1_smooth,_,_ = self.fft_conv(f1,filters=5,width=5,height=5,stride=1,name='f1_smooth')

        print('__debug__spatial_c1',self.spatial_c1)
        return f1
    #
    # def fft_conv(self, source, filters, width, height, stride, activation='relu', name='fft_conv'):
    #     # This function implements a convolution using a spectrally parameterized filter with the normal
    #     # tf.nn.conv2d() convolution operation. This is done by transforming the spectral filter to spatial
    #     # via tf.ifft2d()
    #
    #     channels = source.get_shape().as_list()[3]
    #
    #     with tf.variable_scope(name):
    #         init = self.random_spatial_to_spectral(channels, filters, height, width)
    #         # Option 1: Over-Parameterize fully in the spectral domain
    #         w_real = tf.Variable(init.real, dtype=tf.float32, name='real')
    #         w_imag = tf.Variable(init.imag, dtype=tf.float32, name='imag')
    #         w = tf.cast(tf.complex(w_real, w_imag), tf.complex64)
    #
    #         # Option 2: Parameterize only 'free' parameters in the spectral domain to enforce conjugate symmetry
    #         #		   This is very slow.
    #         # w = self.spectral_to_variable(init)
    #
    #         b = tf.Variable(tf.constant(0.1, shape=[filters]))
    #
    #     # Transform the spectral parameters into a spatial filter
    #     # and reshape for tf.nn.conv2d
    #     complex_spatial_filter = tf.ifft2d(w)
    #     spatial_filter = tf.real(complex_spatial_filter)
    #     spatial_filter = tf.transpose(spatial_filter, [2, 3, 0, 1])
    #
    #     conv = tf.nn.conv2d(source, spatial_filter, strides=[1, stride, stride, 1], padding='SAME')
    #     output = tf.nn.bias_add(conv, b)
    #     output = tf.nn.relu(output) if activation is 'relu' else output
    #
    #     return output, spatial_filter, w

    def fft_conv_pure(self, source, filters, width, height, activation='relu'):
        # This function applies the convolutional filter, which is stored in the spectral domain, as a element-wise
        # multiplication between the filter and the image (which has been transformed to the spectral domain)

        batch_size, input_height, input_width, depth = source.get_shape().as_list()

        # self.sess.run(tf.global_variables_initializer())

        init = self.random_spatial_to_spectral(batch_size, height, width,filters)
        print('shape',init.shape)

            # if name in self.initialization:
            #     init = self.initialization[name]

            # Option 1: Over-Parameterize fully in the spectral domain
        w_real = tf.Variable(init.real, dtype=tf.float32, name='real')
        w_imag = tf.Variable(init.imag, dtype=tf.float32, name='imag')
        w = tf.cast(tf.complex(w_real, w_imag), tf.complex64,name = 'w_complex') # (batch_size,img_width,img_high,c_dim,filter)


        b = tf.Variable(tf.constant(0.1, shape=[filters]))
        print('__debug__w: ',w)
        print('__debug__b: ',b)

        # Add batch as a dimension for later broadcasting
        # w = tf.expand_dims(w, 0)  # batch, channels, filters, height, width
        print(source)
        source = tf.tile(source,[1,1,1,filters])
        print('__debug__source: ', source)

        # Prepare the source tensor for FFT
        # source = tf.transpose(source, [0, , 1, 2])  # batch, channel, height, width
        source_fft = tf.fft2d(tf.complex(source, 0.0 * source))
        print('__debug__source_fft',source_fft)


        conv = source_fft * tf.conj(w)

        # Sum out the channel dimension, and prepare for bias_add
        # Note: The decision to sum out the channel dimension seems intuitive, but
        #	   not necessarily theoretically sound.
        conv = tf.real(tf.ifft2d(conv))
        # conv = tf.reduce_sum(conv, reduction_indices=3)  # batch, filters, height, width
        print('__debug__conv',conv)

        # Drop the batch dimension to keep things consistent with the other conv_op functions
        w = tf.squeeze(w, [0])  # channels, filters, height, width
        w = tf.reduce_sum(w, reduction_indices=2)
        print('__debug__squeeze_w',w)
        # Compute a spatial encoding of the filter for visualization
        spatial_filter = tf.ifft2d(w)


        # Add the bias (in the spatial domain)
        output = tf.nn.bias_add(conv, b)
        output = tf.nn.relu(output) if activation is 'relu' else output
        output = tf.reduce_sum(output, reduction_indices=3)  # batch, filters, height, width

        print('__debug__output',output)
        output = tf.squeeze(output)
        return output, spatial_filter, w



    def random_spatial_to_spectral(self, batch_size, height, width, filters):
        # Create a truncated random image, then compute the FFT of that image and return it's values
        # used to initialize spectrally parameterized filters
        # an alternative to this is to initialize directly in the spectral domain
        w = tf.truncated_normal([batch_size,height,width,filters], mean=0, stddev=0.01)
        fft_ = tf.fft2d(tf.complex(w, 0.0 * w), name='spectral_initializer')
        return fft_.eval(session=self.sess)

    def batch_fftshift2d(self, tensor):
        # Shifts high frequency elements into the center of the filter
        indexes = len(tensor.get_shape()) - 1
        top, bottom = tf.split(tensor, 2, indexes - 1)
        tensor = tf.concat([bottom, top], indexes - 1)
        left, right = tf.split(tensor, 2, indexes)
        tensor = tf.concat([right, left], indexes)

        return tensor

    def batch_ifftshift2d(self, tensor):
        # Shifts high frequency elements into the center of the filter
        indexes = len(tensor.get_shape()) - 1
        left, right = tf.split(tensor, 2, indexes)
        tensor = tf.concat([right, left], indexes)
        top, bottom = tf.split(tensor, 2, indexes - 1)
        tensor = tf.concat([bottom, top], indexes - 1)

        return tensor

    def run(self,hr_img,lr_img):
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        print('run: ->',hr_img.shape)
        # shape = np.zeros(hr_img.shape)
        # err_ = []
        # print(shape)
        for er in range(self.epoch):
            # image = tf.reshape(image,[image.shape[0],image.shape[1]])
            _,x = self.sess.run([self.train_op,self.loss],feed_dict={self.images: lr_img, self.label:hr_img})


            print(x)
        w = self.sess.run([self.spectral_c1],feed_dict={self.images: lr_img, self.label:hr_img})
        w =np.squeeze(w)
        w = w /(1e3*1e-5)
        print(w)
        result = self.pred.eval({self.images: lr_img})
        result = result*255/(1e3*1e-5)
        imshow_spectrum(w)
        imshow(result)
        print(result)