

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf
import numpy as np

import NN_parent as NN

class GoogLeNet (NN.NN):

    # Sources:
    # - Original design: Szegedy et al. (2014)
    # - https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf

    # Note:
    # - the number of filters will be scaled down by a factor of downscale (since we're not identifying the whole set of 1000 classes)

    def __init__ (self, n_classes, img_size=X_shape, tensorboard_verbose=3, name='GoogLeNet', init_model=None, downscale=1, init_stddev=0.01, pretrain_level=100, final_training=True):
        super().__init__(n_classes, img_size, tensorboard_verbose, name, init_model)

        self.img_size = img_size
        self.downscale = downscale
        self.dropout_keepProb = tf.placeholder(tf.float32)
        self.in_nfilter = {}
        self.out_nfilter = {}

        inception_layers = ['11_Conv', '21_Conv', '22_Conv', '31_Conv', '32_Conv', '41_MaxPool', '42_Conv', 'DepthConcat']
        self.layers = {'1_PreNet':['1_Conv', '2_MaxPool', '3_Conv', '4_Conv', '5_MaxPool'], '2_Inception':inception_layers, '3_Inception':inception_layers, '4_Inception':inception_layers, '5_Output':['1_AvgPool', '2_Conv', '3_Dense', '4_Dense'], '5_Inception':inception_layers, '6_Inception':inception_layers, '7_Inception':inception_layers, '8_Inception':inception_layers, '9_Output':['1_AvgPool', '2_Conv', '3_Dense', '4_Dense'], '9_Inception':inception_layers, '10_Inception':inception_layers, '11_Output':['1_AvgPool', '4_Dense']}

        self.layer_pretrain_level = {}
        for layer in self.layers.keys():
            self.layer_pretrain_level[layer] = 0

        self.define_variables(init_stddev=init_stddev, pretrain_level=pretrain_level, final_training=final_training)

    # TODO: adjust to handle meta-levels
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

    def define_conv_variables (self, layer_name, block_name, in_nfilter, out_nfilter, filter_shape, init_stddev=0.01):
        self.in_nfilter[(layer_name, block_name)] = in_nfilter
        self.out_nfilter[(layer_name, block_name)] = out_nfilter
        self.weights[(layer_name, block_name)] = tf.Variable(tf.truncated_normal([*filter_shape, in_nfilter, out_nfilter], stddev=init_stddev), name=(layer_name+'/'+block_name +'-Weights'))
        self.biases[(layer_name, block_name)] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'/'+block_name + '-Biases'))
        return out_nfilter

    def define_dense_variables (self, layer_name, block_name, in_shape, out_shape, init_stddev=0.01):
        self.weights[(layer_name, block_name)] = tf.Variable(tf.truncated_normal([*in_shape, *out_shape], stddev=init_stddev), name=(layer_name+'/'+block_name +'-Weights'))
        self.biases[(layer_name, block_name)] = tf.Variable(tf.zeros([*out_shape,]), name=(layer_name+'/'+block_name + '-Biases'))
        return out_shape

    def define_inception_variables (self, layer_name, in_nfilter, init_stddev=0.01):

        nfilters = []

        with tf.name_scope(layer_name):

            block_name = '11_Conv'
            nfilters.append(self.define_conv_variables(layer_name, block_name, in_nfilter, max(1,int(64/self.downscale)), (1,1)))

            block_name = '21_Conv'
            last_nfilter = self.define_conv_variables(layer_name, block_name, in_nfilter, max(1,int(64/self.downscale)), (1,1))
            block_name = '22_Conv'
            nfilters.append(self.define_conv_variables(layer_name, block_name, last_nfilter, max(1,int(128/self.downscale)), (3,3)))

            block_name = '31_Conv'
            last_nfilter = self.define_conv_variables(layer_name, block_name, in_nfilter, max(1,int(64/self.downscale)), (1,1))
            block_name = '32_Conv'
            nfilters.append(self.define_conv_variables(layer_name, block_name, last_nfilter, max(1,int(32/self.downscale)), (5,5)))

            block_name = '41_MaxPool'
            block_name = '42_Conv'
            nfilters.append(self.define_conv_variables(layer_name, block_name, in_nfilter, max(1,int(64/self.downscale)), (1,1)))

            block_name = 'DepthConcat'

        return np.sum(nfilters)

    def define_variables (self, init_stddev=0.01, pretrain_level=100, final_training=True):

        with tf.name_scope("GoogLeNet") as scope:

            layer_name = "1_PreNet"
            with tf.name_scope(layer_name):

                block_name = "1_Conv"
                last_nfilter = self.define_conv_variables(layer_name, block_name, 1, max(1,int(64/self.downscale)), (7,7))
                
                block_name = "2_MaxPool"
                
                block_name = "3_Conv"
                last_nfilter = self.define_conv_variables(layer_name, block_name, last_nfilter, max(1,int(64/self.downscale)), (1,1))
                
                block_name = "4_Conv"
                last_nfilter = self.define_conv_variables(layer_name, block_name, last_nfilter, max(1,int(192/self.downscale)), (3,3))
                
                block_name = "5_MaxPool"

            layer_name = '2_Inception'
            last_nfilter = self.define_inception_variables(layer_name, last_nfilter)
            layer_name = '3_Inception'
            last_nfilter = self.define_inception_variables(layer_name, last_nfilter)
            layer_name = '4_Inception'
            last_nfilter = self.define_inception_variables(layer_name, last_nfilter)

            layer_name = '5_Output'
            with tf.name_scope(layer_name):
                block_name = '1_AvgPool'
                block_name = '2_Conv'
                last_nfilter_o = self.define_conv_variables(layer_name, block_name, last_nfilter, max(1,int(64/self.downscale)), (1,1))
                block_name = '3_Dense'
                last_shape = self.define_dense_variables(layer_name, block_name, [10,10,last_nfilter_o], [max(1,int(1024/self.downscale)),])
                block_name = '4_Dense'
                last_shape = self.define_dense_variables(layer_name, block_name, last_shape, [2,])

            layer_name = '5_Inception'
            last_nfilter = self.define_inception_variables(layer_name, last_nfilter)
            layer_name = '6_Inception'
            last_nfilter = self.define_inception_variables(layer_name, last_nfilter)
            layer_name = '7_Inception'
            last_nfilter = self.define_inception_variables(layer_name, last_nfilter)
            layer_name = '8_Inception'
            last_nfilter = self.define_inception_variables(layer_name, last_nfilter)

            layer_name = '9_Output'
            with tf.name_scope(layer_name):
                block_name = '1_AvgPool'
                block_name = '2_Conv'
                last_nfilter_o = self.define_conv_variables(layer_name, block_name, last_nfilter, max(1,int(64/self.downscale)), (1,1))
                block_name = '3_Dense'
                last_shape = self.define_dense_variables(layer_name, block_name, [7,7,last_nfilter_o], [max(1,int(1024/self.downscale)),])
                block_name = '4_Dense'
                last_shape = self.define_dense_variables(layer_name, block_name, last_shape, [2,])

            layer_name = '9_Inception'
            last_nfilter = self.define_inception_variables(layer_name, last_nfilter)
            layer_name = '10_Inception'
            last_nfilter = self.define_inception_variables(layer_name, last_nfilter)

            layer_name = '11_Output'
            with tf.name_scope(layer_name):
                block_name = '1_AvgPool'
                block_name = '4_Dense'
                last_shape = self.define_dense_variables(layer_name, block_name, [10,10,last_nfilter], [2,])

    def connect_conv (self, layer_name, block_name, nn, strides, padding='SAME'):
        with tf.name_scope(block_name):
            nn = tf.nn.conv2d(nn,self.weights[(layer_name, block_name)], \
                        strides=strides, padding=padding)
            nn = tf.add(nn, self.biases[(layer_name, block_name)])
            nn = tf.nn.relu(nn)
        return nn

    def connect_inception (self, layer_name, nn):

        with tf.name_scope(layer_name):

            block_name = '11_Conv'
            nn1 = self.connect_conv(layer_name, block_name, nn, strides=[1,1,1,1], padding='SAME')

            block_name = '21_Conv'
            nn2 = self.connect_conv(layer_name, block_name, nn, strides=[1,1,1,1], padding='SAME')
            block_name = '22_Conv'
            nn2 = self.connect_conv(layer_name, block_name, nn2, strides=[1,1,1,1], padding='SAME')

            block_name = '31_Conv'
            nn3 = self.connect_conv(layer_name, block_name, nn, strides=[1,1,1,1], padding='SAME')
            block_name = '32_Conv'
            nn3 = self.connect_conv(layer_name, block_name, nn3, strides=[1,1,1,1], padding='SAME')

            block_name = '41_MaxPool'
            with tf.name_scope(block_name):
                nn4 = tf.nn.max_pool(nn, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
            block_name = '42_Conv'
            nn4 = self.connect_conv(layer_name, block_name, nn4, strides=[1,1,1,1], padding='SAME')

            block_name = 'DepthConcat'
            with tf.name_scope(block_name):
                nn = tf.concat([nn1, nn2, nn3, nn4], 3)

        return nn


    def nn (self, X, pretrain_level=100):

        output_level = pretrain_level

        # build the neural network
        with tf.name_scope("GoogLeNet") as scope:

            if self.img_size != [224,224]:
                layer_name = "ResizeInput"
                with tf.name_scope(layer_name):
                    X = tf.image.resize_images(X, size=[224,224], preserve_aspect_ratio=True)

            nn = X

            layer_name = "1_PreNet"
            with tf.name_scope(layer_name):

                block_name = "1_Conv"
                nn = self.connect_conv(layer_name, block_name, nn, strides=[1,2,2,1])
                
                block_name = "2_MaxPool"
                with tf.name_scope(block_name):
                    nn = tf.nn.max_pool(nn, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
                
                block_name = "3_Conv"
                nn = self.connect_conv(layer_name, block_name, nn, strides=[1,1,1,1])
                
                block_name = "4_Conv"
                nn = self.connect_conv(layer_name, block_name, nn, strides=[1,1,1,1])
                
                block_name = "5_MaxPool"
                with tf.name_scope(block_name):
                    nn = tf.nn.max_pool(nn, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

            layer_name = '2_Inception'
            nn = self.connect_inception(layer_name, nn)
            layer_name = '3_Inception'
            nn = self.connect_inception(layer_name, nn)
            layer_name = '4_Inception'
            nn = self.connect_inception(layer_name, nn)

            if output_level == 0:
                layer_name = '5_Output'
                with tf.name_scope(layer_name):
                    block_name = '1_AvgPool'
                    with tf.name_scope(block_name):
                        nn = tf.nn.avg_pool(nn, ksize=[1,5,5,1], strides=[1,3,3,1], padding='SAME')
                    block_name = '2_Conv'
                    nn = self.connect_conv(layer_name, block_name, nn, strides=[1,1,1,1])
                    block_name = '3_Dense'
                    with tf.name_scope(block_name):
                        nn = tf.tensordot(nn, self.weights[(layer_name, block_name)], axes=[[1,2,3],[0,1,2]])
                        nn = tf.add(nn, self.biases[(layer_name, block_name)])
                        nn = tf.nn.relu(nn)
                        nn = tf.nn.dropout(nn, self.dropout_keepProb)
                    block_name = '4_Dense'
                    with tf.name_scope(block_name):
                        nn = tf.tensordot(nn, self.weights[(layer_name, block_name)], axes=[1,0])
                        nn = tf.add(nn, self.biases[(layer_name, block_name)])
                return nn

            layer_name = '5_Inception'
            nn = self.connect_inception(layer_name, nn)
            layer_name = '6_Inception'
            nn = self.connect_inception(layer_name, nn)
            layer_name = '7_Inception'
            nn = self.connect_inception(layer_name, nn)
            layer_name = '8_Inception'
            nn = self.connect_inception(layer_name, nn)

            if output_level == 1:
                layer_name = '9_Output'
                with tf.name_scope(layer_name):
                    block_name = '1_AvgPool'
                    with tf.name_scope(block_name):
                        nn = tf.nn.avg_pool(nn, ksize=[1,5,5,1], strides=[1,3,3,1], padding='SAME')
                    block_name = '2_Conv'
                    nn = self.connect_conv(layer_name, block_name, nn, strides=[1,1,1,1])
                    block_name = '3_Dense'
                    with tf.name_scope(block_name):
                        nn = tf.tensordot(nn, self.weights[(layer_name, block_name)], axes=[[1,2,3],[0,1,2]])
                        nn = tf.add(nn, self.biases[(layer_name, block_name)])
                        nn = tf.nn.relu(nn)
                        nn = tf.nn.dropout(nn, self.dropout_keepProb)
                    block_name = '4_Dense'
                    with tf.name_scope(block_name):
                        nn = tf.tensordot(nn, self.weights[(layer_name, block_name)], axes=[1,0])
                        nn = tf.add(nn, self.biases[(layer_name, block_name)])
                    return nn

            layer_name = '9_Inception'
            nn = self.connect_inception(layer_name, nn)
            layer_name = '10_Inception'
            nn = self.connect_inception(layer_name, nn)

            layer_name = '11_Output'
            with tf.name_scope(layer_name):
                block_name = '1_AvgPool'
                with tf.name_scope(block_name):
                    nn = tf.nn.avg_pool(nn, ksize=[1,5,5,1], strides=[1,3,3,1], padding='SAME')
                block_name = '4_Dense'
                with tf.name_scope(block_name):
                    nn = tf.tensordot(nn, self.weights[(layer_name, block_name)], axes=[[1,2,3],[0,1,2]])
                    nn = tf.add(nn, self.biases[(layer_name, block_name)])

            return nn