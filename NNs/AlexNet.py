

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf

import NN_parent as NN

class AlexNet (NN.NN):

    # Sources:
    # - Original design: Krizhevsky, Sutskever, and Hinton (2012)
    # - https://software.intel.com/en-us/articles/hands-on-ai-part-16-modern-deep-neural-network-architectures-for-image-classification
    # - http://cvml.ist.ac.at/courses/DLWT_W17/material/AlexNet.pdf

    # Note:
    # - the number of filters will be scaled down by a factor of downscale (since we're not identifying the whole set of 1000 classes)

    def __init__ (self, n_classes, img_size=X_shape, tensorboard_verbose=3, name='AlexNet', init_model=None, downscale=1, init_stddev=0.02):
        super().__init__(n_classes, img_size, tensorboard_verbose, name, init_model)

        self.img_size = img_size
        self.downscale = downscale
        self.dropout_keepProb = tf.placeholder(tf.float32)

        self.define_variables(init_stddev=init_stddev)

    def define_variables (self, init_stddev=0.02):

        with tf.name_scope("AlexNet") as scope:

            layer_name = "ConvPool1"; self.layers.append(layer_name)
            in_nfilter = 1; out_nfilter = max(1,int(48/self.downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([11, 11,in_nfilter, out_nfilter], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'-Biases'))

            layer_name = "ConvPool2"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(128/self.downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([5,5,in_nfilter, out_nfilter], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'-Biases'))

            layer_name = "Conv1"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(192/self.downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,in_nfilter, out_nfilter], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'-Biases'))

            layer_name = "Conv2"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(192/self.downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,in_nfilter, out_nfilter], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'-Biases'))

            layer_name = "ConvPool3"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(128/self.downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,in_nfilter, out_nfilter], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'-Biases'))

            layer_name = "Dense1"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(2048/self.downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([14,14, in_nfilter, out_nfilter], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'-Biases'))

            layer_name = "Dense2"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(2048/self.downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([in_nfilter, out_nfilter], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'-Biases'))

            layer_name = "Output"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = self.n_classes
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([in_nfilter, out_nfilter], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'-Biases'))


    def nn (self, X):

        # build the neural network
        with tf.name_scope("AlexNet") as scope:

            if self.img_size != [224,224]:
                layer_name = "ResizeInput"; self.layers.append(layer_name)
                with tf.name_scope(layer_name):
                    X = tf.image.resize_images(X, size=[224,224], preserve_aspect_ratio=True)

            layer_name = "ConvPool1"; self.layers.append(layer_name)
            in_nfilter = 1; out_nfilter = max(1,int(48/self.downscale))
            with tf.name_scope(layer_name):

                _nn = tf.nn.conv2d(X,self.weights[layer_name], \
                    strides=[1,4,4,1], padding='SAME', name=(layer_name+'-conv2d'))
                self.firstConv = X
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.max_pool(_nn, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                _nn = tf.nn.relu(_nn)
                # normalization step ignored

            layer_name = "ConvPool2"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(128/self.downscale))
            with tf.name_scope(layer_name):

                _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                    strides=[1,4,4,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.max_pool(_nn, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                _nn = tf.nn.relu(_nn)
                # normalization step ignored

            layer_name = "Conv1"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(192/self.downscale))
            with tf.name_scope(layer_name):

                _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                    strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)

            layer_name = "Conv2"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(192/self.downscale))
            with tf.name_scope(layer_name):

                _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                    strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)

            layer_name = "ConvPool3"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(128/self.downscale))
            with tf.name_scope(layer_name):

                _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                    strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.max_pool(_nn, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                _nn = tf.nn.relu(_nn)

            layer_name = "Dense1"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(2048/self.downscale))
            with tf.name_scope(layer_name):

                _nn = tf.nn.dropout(_nn, self.dropout_keepProb)
                _nn = tf.tensordot(_nn, self.weights[layer_name], axes=[[1,2,3],[0,1,2]])
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)

            layer_name = "Dense2"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(2048/self.downscale))
            with tf.name_scope(layer_name):

                _nn = tf.nn.dropout(_nn, self.dropout_keepProb)
                _nn = tf.matmul(_nn, self.weights[layer_name])
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)

            layer_name = "Output"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = self.n_classes
            with tf.name_scope(layer_name):

                _nn = tf.matmul(_nn, self.weights[layer_name])
                _nn = tf.add(_nn, self.biases[layer_name])

            return _nn