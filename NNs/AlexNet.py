

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

    def __init__ (self, n_classes, img_size=X_shape, tensorboard_verbose=3, name='AlexNet', init_model=None, downscale=1):
        super().__init__(n_classes, img_size, tensorboard_verbose, name, init_model)

        self.downscale = downscale

        # build the neural network
        with tf.name_scope("AlexNet") as scope:

            if img_size != [224,224]:
                layer_name = "ResizeInput"; self.layers.append(layer_name)
                with tf.name_scope(layer_name):
                    self.X = tf.image.resize_images(self.X, size=[224,224], preserve_aspect_ratio=True)

            layer_name = "ConvPool1"; self.layers.append(layer_name)
            in_nfilter = 1; out_nfilter = max(1,int(48/downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([*X_shape,in_nfilter, out_nfilter], stddev=0.02), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'-Biases'))
                _nn = tf.nn.conv2d(self.X,self.weights[layer_name], \
                    strides=[1,4,4,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.max_pool(_nn, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                _nn = tf.nn.relu(_nn)
                # normalization step ignored

            layer_name = "ConvPool2"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(128/downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([5,5,in_nfilter, out_nfilter], stddev=0.02), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'-Biases'))
                _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                    strides=[1,4,4,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.max_pool(_nn, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                _nn = tf.nn.relu(_nn)
                # normalization step ignored

            layer_name = "Conv1"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(192/downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,in_nfilter, out_nfilter], stddev=0.02), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'-Biases'))
                _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                    strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)

            layer_name = "Conv2"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(192/downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,in_nfilter, out_nfilter], stddev=0.02), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'-Biases'))
                _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                    strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)

            layer_name = "ConvPool3"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(128/downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,in_nfilter, out_nfilter], stddev=0.02), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'-Biases'))
                _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                    strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.max_pool(_nn, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                _nn = tf.nn.relu(_nn)

            layer_name = "Dense1"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(2048/downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([14,14, in_nfilter, out_nfilter], stddev=0.02), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'-Biases'))

                _nn = tf.tensordot(_nn, self.weights[layer_name], axes=[[1,2,3],[0,1,2]])
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)

            layer_name = "Dense2"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = max(1,int(2048/downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([in_nfilter, out_nfilter], stddev=0.02), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'-Biases'))

                _nn = tf.matmul(_nn, self.weights[layer_name])
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)

            layer_name = "Output"; self.layers.append(layer_name)
            in_nfilter = out_nfilter; out_nfilter = self.n_classes
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([in_nfilter, out_nfilter], stddev=0.02), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([out_nfilter,]), name=(layer_name+'-Biases'))

                _nn = tf.matmul(_nn, self.weights[layer_name])
                _nn = tf.add(_nn, self.biases[layer_name])

            self.nn = _nn