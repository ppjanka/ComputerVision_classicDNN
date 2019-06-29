

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf
import numpy as np

import NN_parent as NN

class VGGNet (NN.NN):

    # Sources:
    # - Original design: Zeiler & Fergus (2013)
    # - https://software.intel.com/en-us/articles/hands-on-ai-part-16-modern-deep-neural-network-architectures-for-image-classification

    # Note:
    # - the number of filters will be scaled down by a factor of downscale (since we're not identifying the whole set of 1000 classes)

    def __init__ (self, n_classes, img_size=X_shape, tensorboard_verbose=3, name='VGGNet', init_model=None, downscale=1, init_stddev=0.01, pretrain_level=100, final_training=True):
        super().__init__(n_classes, img_size, tensorboard_verbose, name, init_model)

        self.img_size = img_size
        self.downscale = downscale
        self.dropout_keepProb = tf.placeholder(tf.float32)
        self.in_nfilter = {}
        self.out_nfilter = {}

        self.layers = ['1_Conv', '2_Conv', '3_MaxPool', '4_Conv', '5_Conv', '6_MaxPool', '7_Conv', '8_Conv', '9_Conv1', '9_Conv3', '10_MaxPool', '11_Conv', '12_Conv', '13_Conv1', '13_Conv3', '14_MaxPool', '15_Conv', '16_Conv', '17_Conv1', '17_Conv3', '18_MaxPool', '19_Dense', '20_Dense', '21_Output']

        self.layer_pretrain_level = {}
        self.layer_pretrain_level['1_Conv'] = 0
        self.layer_pretrain_level['2_Conv'] = 1
        self.layer_pretrain_level['3_MaxPool'] = 0
        self.layer_pretrain_level['4_Conv'] = 0
        self.layer_pretrain_level['5_Conv'] = 2
        self.layer_pretrain_level['6_MaxPool'] = 0
        self.layer_pretrain_level['7_Conv'] = 0
        self.layer_pretrain_level['8_Conv'] = 3
        self.layer_pretrain_level['9_Conv1'] = 100
        self.layer_pretrain_level['9_Conv3'] = 7
        self.layer_pretrain_level['10_MaxPool'] = 0
        self.layer_pretrain_level['11_Conv'] = 0
        self.layer_pretrain_level['12_Conv'] = 4
        self.layer_pretrain_level['13_Conv1'] = 100
        self.layer_pretrain_level['13_Conv3'] = 8
        self.layer_pretrain_level['14_MaxPool'] = 0
        self.layer_pretrain_level['15_Conv'] = 0
        self.layer_pretrain_level['16_Conv'] = 5
        self.layer_pretrain_level['17_Conv1'] = 100
        self.layer_pretrain_level['17_Conv3'] = 9
        self.layer_pretrain_level['18_MaxPool'] = 0
        self.layer_pretrain_level['19_Dense'] = 0
        self.layer_pretrain_level['20_Dense'] = 6
        self.layer_pretrain_level['21_Output'] = 0

        self.define_variables(init_stddev=init_stddev, pretrain_level=pretrain_level, final_training=final_training)

    # copy weights from pretrained levels to the ones to be trained -- might be easier to train this way
    def init_pretrain_level (self, sess, pretrain_level):
        if pretrain_level > 0:
            print('Initializing layers for pretraining')
            to_be_initialized = np.array(list(self.layer_pretrain_level.keys()))[np.array(list(self.layer_pretrain_level.values())) == pretrain_level]
            for layer in to_be_initialized:
                if layer == '20_Dense': continue
                if layer in self.weights.keys() and layer in self.biases.keys():
                    init_layer = self.layers[self.layers.index(layer)-1]
                    print(' - initializing layer %s with %s' % (layer, init_layer))
                    init_shape = tf.shape(self.weights[init_layer])
                    dest_shape = tf.shape(self.weights[layer])
                    multiples = tf.cast(dest_shape / init_shape, tf.int32)
                    init_buffer = tf.tile(self.weights[init_layer], multiples=multiples)
                    sess.run(self.weights[layer].assign(init_buffer))
                    init_shape = tf.shape(self.biases[init_layer])
                    dest_shape = tf.shape(self.biases[layer])
                    multiples = tf.cast(dest_shape / init_shape, tf.int32)
                    init_buffer = tf.tile(self.biases[init_layer], multiples=multiples)
                    sess.run(self.biases[layer].assign(init_buffer))
            print('done.')


    def define_variables (self, init_stddev=0.01, pretrain_level=100, final_training=True):

        with tf.name_scope("VGGNet") as scope:

            layer_name = "1_Conv"
            self.in_nfilter[layer_name] = 1
            self.out_nfilter[layer_name] = max(1,int(64/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            if False:# pretrain_level > self.layer_pretrain_level[layer_name] and not final_training:
                self.weights[layer_name] = tf.stop_gradient(self.weights[layer_name])
                self.biases[layer_name] = tf.stop_gradient(self.biases[layer_name])
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "2_Conv"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(64/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            if False:# pretrain_level > self.layer_pretrain_level[layer_name] and not final_training:
                self.weights[layer_name] = tf.stop_gradient(self.weights[layer_name])
                self.biases[layer_name] = tf.stop_gradient(self.biases[layer_name])
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "3_MaxPool"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(64/self.downscale))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "4_Conv"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(128/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            if False:# pretrain_level > self.layer_pretrain_level[layer_name] and not final_training:
                self.weights[layer_name] = tf.stop_gradient(self.weights[layer_name])
                self.biases[layer_name] = tf.stop_gradient(self.biases[layer_name])
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "5_Conv"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(128/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "6_MaxPool"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(128/self.downscale))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "7_Conv"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(256/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            if False:# pretrain_level > self.layer_pretrain_level[layer_name] and not final_training:
                self.weights[layer_name] = tf.stop_gradient(self.weights[layer_name])
                self.biases[layer_name] = tf.stop_gradient(self.biases[layer_name])
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "8_Conv"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(256/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "9_Conv1"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(256/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([1,1,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "9_Conv3"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(256/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "10_MaxPool"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(256/self.downscale))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "11_Conv"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            if False:#pretrain_level > self.layer_pretrain_level[layer_name] and not final_training:
                self.weights[layer_name] = tf.stop_gradient(self.weights[layer_name])
                self.biases[layer_name] = tf.stop_gradient(self.biases[layer_name])
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "12_Conv"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "13_Conv1"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([1,1,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "13_Conv3"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "14_MaxPool"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "15_Conv"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            if False:# pretrain_level > self.layer_pretrain_level[layer_name] and not final_training:
                self.weights[layer_name] = tf.stop_gradient(self.weights[layer_name])
                self.biases[layer_name] = tf.stop_gradient(self.biases[layer_name])
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "16_Conv"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "17_Conv1"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([1,1,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "17_Conv3"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "18_MaxPool"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "19_Dense"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(4096/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([7,7, self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            if False:# pretrain_level > self.layer_pretrain_level[layer_name] and not final_training:
                self.weights[layer_name] = tf.stop_gradient(self.weights[layer_name])
                self.biases[layer_name] = tf.stop_gradient(self.biases[layer_name])
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "20_Dense"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(4096/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "21_Output"
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = self.n_classes
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))


    def nn (self, X, pretrain_level=100):

        # build the neural network
        with tf.name_scope("VGGNet") as scope:

            _subnn = {}

            if self.img_size != [224,224]:
                layer_name = "ResizeInput"
                with tf.name_scope(layer_name):
                    X = tf.image.resize_images(X, size=[224,224], preserve_aspect_ratio=True)

            if False: # expensive
                mean = tf.reduce_mean(X, axis=[1,2,3])
                Xmin = tf.reduce_min(X)
                Xmax = tf.reduce_max(X)
                X = (X-Xmin)/(Xmax-Xmin) - 0.5

            layer_name = "1_Conv"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(X,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "2_Conv"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "3_MaxPool"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.max_pool(_nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                _subnn[layer_name] = _nn

            layer_name = "4_Conv"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "5_Conv"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "6_MaxPool"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.max_pool(_nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                _subnn[layer_name] = _nn

            layer_name = "7_Conv"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "8_Conv"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "9_Conv1"
            if pretrain_level == self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "9_Conv3"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "10_MaxPool"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.max_pool(_nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                _subnn[layer_name] = _nn

            layer_name = "11_Conv"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "12_Conv"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "13_Conv1"
            if pretrain_level == self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "13_Conv3"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "14_MaxPool"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.max_pool(_nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                _subnn[layer_name] = _nn

            layer_name = "15_Conv"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "16_Conv"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "17_Conv1"
            if pretrain_level == self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "17_Conv3"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "18_MaxPool"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.nn.max_pool(_nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                _subnn[layer_name] = _nn

            layer_name = "19_Dense"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.tensordot(_nn, self.weights[layer_name], axes=[[1,2,3],[0,1,2]])
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                    _nn = tf.nn.dropout(_nn, self.dropout_keepProb)
                _subnn[layer_name] = _nn

            layer_name = "20_Dense"
            if pretrain_level >= self.layer_pretrain_level[layer_name]:
                with tf.name_scope(layer_name):
                    _nn = tf.tensordot(_nn, self.weights[layer_name], axes=[1,0])
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                    _nn = tf.nn.dropout(_nn, self.dropout_keepProb)
                _subnn[layer_name] = _nn

            layer_name = "21_Output"
            with tf.name_scope(layer_name):
                _nn = tf.matmul(_nn, self.weights[layer_name])
                _nn = tf.add(_nn, self.biases[layer_name])
            _subnn[layer_name] = _nn

            return _nn