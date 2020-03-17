

import tensorflow as tf
import numpy as np
from utils import imshow

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


        # self.images = tf.reshape(self.images,[1, 256, 256, 1])
        # self.label = tf.reshape(self.label, [1, 256, 256, 1])
        # w1= tf.placeholder(tf.random_normal([None, None], stddev=1e-3), name='w1'),

        #self.w1 = tf.placeholder(tf.float32,[None,None], name='w_1')
        #self.w1 = self.w1 + tf.random_normal(tf.shape(self.w1))

        self.weights = {
            'w1': tf.Variable(tf.random_normal([256, 256],  stddev=1e-3), name='w1'),
            'w2': tf.Variable(tf.random_normal([256, 256], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal([256, 256], stddev=1e-3), name='w3'),
            'w4': tf.Variable(tf.random_normal([256, 256], stddev=1e-3), name='w4'),
            'w5': tf.Variable(tf.random_normal([256, 256], stddev=1e-3), name='w5')
        }

        self.biases = {
            'b1': tf.Variable(tf.zeros([256, 256], name='b1')),
            'b2': tf.Variable(tf.zeros([256, 256], name='b2')),
            'b3': tf.Variable(tf.zeros([256, 256], name='b3')),
            'b4': tf.Variable(tf.zeros([256, 256], name='b4')),
            'b5': tf.Variable(tf.zeros([256, 256], name='b5'))

        }

        self.smooth = {
            's1': tf.Variable(tf.random_normal([5, 5,1,1], stddev=1e-3), name='s1'),
            's2': tf.Variable(tf.random_normal([5,5,1,1], stddev=1e-3), name='s2'),
            's3': tf.Variable(tf.random_normal([5,5,1,1], stddev=1e-3), name='s3'),
            's4': tf.Variable(tf.random_normal([5,5,1,1], stddev=1e-3), name='s4'),
            's5': tf.Variable(tf.random_normal([5,5,1,1], stddev=1e-3), name='s5')
        }


        self.pred = self.model()
        self.loss = tf.nn.l2_loss(self.label - self.pred)
        # print(self.pred)
        # self.loss = tf.reduce_mean(tf.square(self.label - self.pred))
        # print('build_model_image_shape',self.images)

    def conv_(self,x,weights,biases,smooth):


        # x1 = (tf.math.multiply(x, weights) + biases)
        # x1 = tf.reshape(x1,[1,x1.shape[0],x1.shape[1],1])
        conv1 = (tf.nn.conv2d(x, smooth, strides=[1,1,1,1], padding='SAME'))

        #
        # x2 = (tf.math.multiply(x, weights) + biases)
        # x2 = tf.reshape(x2,[1,x2.shape[0],x2.shape[1],1])
        # conv2 = (tf.nn.conv2d(x2, smooth, strides=[1,1,1,1], padding='SAME'))
        #
        # x3 = (tf.math.multiply(x, weights) + biases)
        # x3 = tf.reshape(x3,[1,x3.shape[0],x3.shape[1],1])
        # conv3 = (tf.nn.conv2d(x3, smooth, strides=[1,1,1,1], padding='SAME'))
        #
        # x4 = (tf.math.multiply(x, weights) + biases)
        # x4 = tf.reshape(x4,[1,x4.shape[0],x4.shape[1],1])
        # conv4 = (tf.nn.conv2d(x4, smooth, strides=[1,1,1,1], padding='SAME'))
        #
        # x5 = (tf.math.multiply(x, weights) + biases)
        # x5 = tf.reshape(x5,[1,x5.shape[0],x5.shape[1],1])
        # conv5 = (tf.nn.conv2d(x5, smooth, strides=[1,1,1,1], padding='SAME'))
        #
        # # x_out = tf.reduce_sum([conv1,conv2,conv3,conv4,conv5])
        # x_out = conv1 + conv2 + conv3 + conv4 + conv5
        x_out = conv1
        # print('debug: ',x)
        print(x_out)
        return tf.squeeze(x_out)



    def model(self):
        # x = None
        f1 = self.conv_(self.images,self.weights['w1'],self.biases['b1'],self.smooth['s1'])
        f2 = self.conv_(f1, self.weights['w2'],self.biases['b2'],self.smooth['s2'])
        f3 = self.conv_(f2, self.weights['w3'],self.biases['b3'],self.smooth['s3'])
        f4 = self.conv_(f3, self.weights['w4'],self.biases['b4'],self.smooth['s4'])
        f5 = self.conv_(f4,self.weights['w5'],self.biases['b5'],self.smooth['s5'])
        # f6 = self.conv_(f5, self.weights['w6'],self.biases['b1'],self.smooth['s1'])

        # print("debug ->",f1)
        fout = f1+f2+f3+f4+f5
        # fout = f1
        # fout = tf.transpose(fout)
        p = fout * self.weights['w5']
        I_star = p+f1
        return I_star

    def run(self,hr_img,lr_img):
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        tf.initialize_all_variables().run()
        print('run: ->',hr_img.shape)
        # shape = np.zeros(hr_img.shape)
        # err_ = []
        # print(shape)
        for er in range(self.epoch):
            # image = tf.reshape(image,[image.shape[0],image.shape[1]])
            _,x = self.sess.run([self.train_op,self.loss],feed_dict={self.images: lr_img, self.label:hr_img})
            # w = self.sess.run([self.w1],feed_dict={self.w1:shape})
            # print(w)

            # imshow(x_out)
            print(x)
        # return x_out
        #imshow()
            # print(t)