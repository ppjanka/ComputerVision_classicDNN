

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf

import NN_parent as NN

class ZFNet (NN.NN):

    # Sources:
    # - Original design: Zeiler & Fergus (2013)
    # - https://software.intel.com/en-us/articles/hands-on-ai-part-16-modern-deep-neural-network-architectures-for-image-classification

    # Note:
    # - the number of filters will be scaled down by a factor of downscale (since we're not identifying the whole set of 1000 classes)

    def __init__ (self, n_classes, img_size=X_shape, tensorboard_verbose=3, name='ZFNet', init_model=None, downscale=1, init_stddev=0.02):
        super().__init__(n_classes, img_size, tensorboard_verbose, name, init_model)

        self.img_size = img_size
        self.downscale = downscale
        self.dropout_keepProb = tf.placeholder(tf.float32)
        self.in_nfilter = {}
        self.out_nfilter = {}

        self.define_variables(init_stddev=init_stddev)

    def define_variables (self, init_stddev=0.02):

        with tf.name_scope("ZFNet") as scope:

            layer_name = "ConvPool1"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = 1
            self.out_nfilter[layer_name] = max(1,int(96/self.downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([7,7,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))

            layer_name = "ConvPool2"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = self.out_nfilter["ConvPool1"]
            self.out_nfilter[layer_name] = max(1,int(256/self.downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([5,5,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))

            layer_name = "Conv1"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = self.out_nfilter["ConvPool2"]
            self.out_nfilter[layer_name] = max(1,int(384/self.downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))

            layer_name = "Conv2"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = self.out_nfilter["Conv1"]
            self.out_nfilter[layer_name] = max(1,int(384/self.downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))

            layer_name = "ConvPool3"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = self.out_nfilter["Conv2"]
            self.out_nfilter[layer_name] = max(1,int(256/self.downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))

            layer_name = "Dense1"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = self.out_nfilter["ConvPool3"]
            self.out_nfilter[layer_name] = max(1,int(4096/self.downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([7,7, self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))

            layer_name = "Dense2"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = self.out_nfilter["Dense1"]
            self.out_nfilter[layer_name] = max(1,int(4096/self.downscale))
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))

            layer_name = "Output"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = self.out_nfilter["Dense2"]
            self.out_nfilter[layer_name] = self.n_classes
            with tf.name_scope(layer_name):

                self.weights[layer_name] = tf.Variable(tf.truncated_normal([self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))


    def nn (self, X):

        # build the neural network
        with tf.name_scope("ZFNet") as scope:

            if self.img_size != [224,224]:
                layer_name = "ResizeInput"; self.layers.append(layer_name)
                with tf.name_scope(layer_name):
                    X = tf.image.resize_images(X, size=[224,224], preserve_aspect_ratio=True)

            layer_name = "ConvPool1"
            with tf.name_scope(layer_name):

                _nn = tf.nn.conv2d(X,self.weights[layer_name], \
                    strides=[1,2,2,1], padding='SAME', name=(layer_name+'-conv2d'))
                self.firstConv = X
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.max_pool(_nn, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
                _nn = tf.nn.relu(_nn)
                # normalization step ignored

            layer_name = "ConvPool2"
            with tf.name_scope(layer_name):

                _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                    strides=[1,2,2,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.max_pool(_nn, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
                _nn = tf.nn.relu(_nn)
                # normalization step ignored

            layer_name = "Conv1"
            with tf.name_scope(layer_name):

                _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                    strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)

            layer_name = "Conv2"
            with tf.name_scope(layer_name):

                _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                    strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)

            layer_name = "ConvPool3"
            with tf.name_scope(layer_name):

                _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                    strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.max_pool(_nn, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
                _nn = tf.nn.relu(_nn)

            layer_name = "Dense1"
            with tf.name_scope(layer_name):

                _nn = tf.nn.dropout(_nn, self.dropout_keepProb)
                _nn = tf.tensordot(_nn, self.weights[layer_name], axes=[[1,2,3],[0,1,2]])
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)

            layer_name = "Dense2"
            with tf.name_scope(layer_name):

                _nn = tf.nn.dropout(_nn, self.dropout_keepProb)
                _nn = tf.matmul(_nn, self.weights[layer_name])
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)

            layer_name = "Output"
            with tf.name_scope(layer_name):

                _nn = tf.matmul(_nn, self.weights[layer_name])
                _nn = tf.add(_nn, self.biases[layer_name])

            return _nn