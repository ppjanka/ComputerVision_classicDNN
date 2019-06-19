

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf

import NN_parent as NN

class VGGNet (NN.NN):

    # Sources:
    # - Original design: Zeiler & Fergus (2013)
    # - https://software.intel.com/en-us/articles/hands-on-ai-part-16-modern-deep-neural-network-architectures-for-image-classification

    # Note:
    # - the number of filters will be scaled down by a factor of downscale (since we're not identifying the whole set of 1000 classes)

    def __init__ (self, n_classes, img_size=X_shape, tensorboard_verbose=3, name='VGGNet', init_model=None, downscale=1, init_stddev=0.01):
        super().__init__(n_classes, img_size, tensorboard_verbose, name, init_model)

        self.img_size = img_size
        self.downscale = downscale
        self.dropout_keepProb = tf.placeholder(tf.float32)
        self.in_nfilter = {}
        self.out_nfilter = {}

        self.define_variables(init_stddev=init_stddev)

    def define_variables (self, init_stddev=0.01):

        with tf.name_scope("VGGNet") as scope:

            layer_name = "1_Conv"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = 1
            self.out_nfilter[layer_name] = max(1,int(64/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "2_Conv"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(64/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "3_MaxPool"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(64/self.downscale))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "4_Conv"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(128/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "5_Conv"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(128/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "6_MaxPool"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(128/self.downscale))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "7_Conv"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(256/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "8_Conv"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(256/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "9_Conv1"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(256/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([1,1,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "9_Conv3"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(256/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "10_MaxPool"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(256/self.downscale))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "11_Conv"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "12_Conv"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "13_Conv1"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([1,1,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "13_Conv3"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "14_MaxPool"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "15_Conv"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "16_Conv"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "17_Conv1"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([1,1,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "17_Conv3"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([3,3,self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "18_MaxPool"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(512/self.downscale))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "19_Dense"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(4096/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([7,7, self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "20_Dense"; self.layers.append(layer_name)
            self.in_nfilter[layer_name] = prev_nfilter
            self.out_nfilter[layer_name] = max(1,int(4096/self.downscale))
            with tf.name_scope(layer_name):
                self.weights[layer_name] = tf.Variable(tf.truncated_normal([self.in_nfilter[layer_name], self.out_nfilter[layer_name]], stddev=init_stddev), name=(layer_name+'-Weights'))
                self.biases[layer_name] = tf.Variable(tf.zeros([self.out_nfilter[layer_name],]), name=(layer_name+'-Biases'))
            prev_nfilter = self.out_nfilter[layer_name]

            layer_name = "21_Output"; self.layers.append(layer_name)
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
                layer_name = "ResizeInput"; self.layers.append(layer_name)
                with tf.name_scope(layer_name):
                    X = tf.image.resize_images(X, size=[224,224], preserve_aspect_ratio=True)

            if False: # expensive
                mean = tf.reduce_mean(X, axis=[1,2,3])
                Xmin = tf.reduce_min(X)
                Xmax = tf.reduce_max(X)
                X = (X-Xmin)/(Xmax-Xmin) - 0.5

            layer_name = "1_Conv"
            with tf.name_scope(layer_name):
                _nn = tf.nn.conv2d(X,self.weights[layer_name], \
                    strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)
            _subnn[layer_name] = _nn

            if pretrain_level > 0:
                layer_name = "2_Conv"
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "3_MaxPool"
            with tf.name_scope(layer_name):
                _nn = tf.nn.max_pool(_nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            _subnn[layer_name] = _nn

            layer_name = "4_Conv"
            with tf.name_scope(layer_name):
                _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                    strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)
            _subnn[layer_name] = _nn

            if pretrain_level > 0:
                layer_name = "5_Conv"
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            layer_name = "6_MaxPool"
            with tf.name_scope(layer_name):
                _nn = tf.nn.max_pool(_nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            _subnn[layer_name] = _nn

            layer_name = "7_Conv"
            with tf.name_scope(layer_name):
                _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                    strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)
            _subnn[layer_name] = _nn

            if pretrain_level > 0:
                layer_name = "8_Conv"
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            if pretrain_level > 1:

                if pretrain_level < 3:
                    _nn = _subnn["8_Conv"]
                    layer_name = "9_Conv1"
                    with tf.name_scope(layer_name):
                        _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                            strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                        _nn = tf.add(_nn, self.biases[layer_name])
                        _nn = tf.nn.relu(_nn)
                    _subnn[layer_name] = _nn

                if pretrain_level > 2:
                    _nn = _subnn["8_Conv"]
                    layer_name = "9_Conv3"
                    with tf.name_scope(layer_name):
                        _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                            strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                        _nn = tf.add(_nn, self.biases[layer_name])
                        _nn = tf.nn.relu(_nn)
                    _subnn[layer_name] = _nn

            layer_name = "10_MaxPool"
            with tf.name_scope(layer_name):
                _nn = tf.nn.max_pool(_nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            _subnn[layer_name] = _nn

            layer_name = "11_Conv"
            with tf.name_scope(layer_name):
                _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                    strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)
            _subnn[layer_name] = _nn

            if pretrain_level > 0:
                layer_name = "12_Conv"
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            if pretrain_level > 1:

                if pretrain_level < 3:
                    _nn = _subnn['12_Conv']
                    layer_name = "13_Conv1"
                    with tf.name_scope(layer_name):
                        _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                            strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                        _nn = tf.add(_nn, self.biases[layer_name])
                        _nn = tf.nn.relu(_nn)
                    _subnn[layer_name] = _nn

                if pretrain_level > 2:
                    _nn = _subnn['12_Conv']
                    layer_name = "13_Conv3"
                    with tf.name_scope(layer_name):
                        _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                            strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                        _nn = tf.add(_nn, self.biases[layer_name])
                        _nn = tf.nn.relu(_nn)
                    _subnn[layer_name] = _nn

            layer_name = "14_MaxPool"
            with tf.name_scope(layer_name):
                _nn = tf.nn.max_pool(_nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            _subnn[layer_name] = _nn

            layer_name = "15_Conv"
            with tf.name_scope(layer_name):
                _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                    strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)
            _subnn[layer_name] = _nn

            if pretrain_level > 0:
                layer_name = "16_Conv"
                with tf.name_scope(layer_name):
                    _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                        strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                    _nn = tf.add(_nn, self.biases[layer_name])
                    _nn = tf.nn.relu(_nn)
                _subnn[layer_name] = _nn

            if pretrain_level > 1:

                if pretrain_level < 3:
                    _nn = _subnn['16_Conv']
                    layer_name = "17_Conv1"
                    with tf.name_scope(layer_name):
                        _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                            strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                        _nn = tf.add(_nn, self.biases[layer_name])
                        _nn = tf.nn.relu(_nn)
                    _subnn[layer_name] = _nn

                if pretrain_level > 2:
                    _nn = _subnn['16_Conv']
                    layer_name = "17_Conv3"
                    with tf.name_scope(layer_name):
                        _nn = tf.nn.conv2d(_nn,self.weights[layer_name], \
                            strides=[1,1,1,1], padding='SAME', name=(layer_name+'-conv2d'))
                        _nn = tf.add(_nn, self.biases[layer_name])
                        _nn = tf.nn.relu(_nn)
                    _subnn[layer_name] = _nn

            layer_name = "18_MaxPool"
            with tf.name_scope(layer_name):
                _nn = tf.nn.max_pool(_nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            _subnn[layer_name] = _nn

            layer_name = "19_Dense"
            with tf.name_scope(layer_name):
                _nn = tf.tensordot(_nn, self.weights[layer_name], axes=[[1,2,3],[0,1,2]])
                _nn = tf.add(_nn, self.biases[layer_name])
                _nn = tf.nn.relu(_nn)
                _nn = tf.nn.dropout(_nn, self.dropout_keepProb)
            _subnn[layer_name] = _nn

            if pretrain_level > 1:
                layer_name = "20_Dense"
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