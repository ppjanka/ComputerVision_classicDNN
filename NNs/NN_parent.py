

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf
import numpy as np

import cv_io as cvio
import cv_train_val as cvtrain

class NN:

    def __init__ (self, n_classes, img_size=X_shape, tensorboard_verbose=3, name='NN', init_model=None):

        self.name = name
        self.last_model = init_model

        # "placeholders" to be replaced by data tensors
        self.X = tf.ones(dtype=tf.float32, shape=[batch_size, *img_size, 1], name='INPUT')
        self.y = tf.ones(dtype=tf.float32, shape=[batch_size, n_classes, 1], name='LABELS')

        self.n_classes = n_classes

        self.layers = []
        self.weights = {}
        self.biases = {}

        self.nn = None
      
    def predict (self, Xin, model=None):
        if model == None:
            model = self.last_model
          
        graph = self.nn
        saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
            saver.restore(sess, model)
            result = graph.eval({self.X:Xin, self.keep_prob:1.})
        return result

    def train (self, preprocessed_folder='./data_preprocessed', validation_size=0.2, n_epoch=1, input_model='last', output_model=None, logfile=None):

        # initialize
        tf.logging.set_verbosity(tf.logging.INFO)
        last_epoch = tf.Variable(-1, name='last_epoch')
        saver = tf.train.Saver()

        # input
        import os
        if not os.path.isdir(preprocessed_folder):
            print(preprocessed_folder + ' does not exist. Please preprocess data using preprocess.py first.')
            return 0
        data = cvio.image_batch_handle(preprocessed_folder, validation_size=validation_size)

        # setup training and validation handles
        optimizer, train_cost, train_acc = cvtrain.training_handle (data['train']['y'], self.nn)
        val_cost, val_acc = cvtrain.validation_handle(data['val']['y'], self.nn)

        with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
              
            writer = tf.summary.FileWriter('./workspace/log_%s/1/train' % self.name, sess.graph)

            print('Initializing all variables.. ', end='', flush=True)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            print('done.', flush=True)

            print('Initializing logfile.. ', end='', flush=True)
            if logfile == None:
                logfile = './workspace/%s.log' % (self.name,)
            if not os.path.isfile(logfile):
                with open(logfile, 'a') as f:
                    f.write('Epoch\tAvgTrainLoss\tAvgTrainAcc\tAvgValLoss\tAvgValAcc\n')
            print('done.', flush=True)

            # restore model if specified
            if input_model == 'last' and self.last_model != None:
                print('Restoring the last available model:\n  %s' % self.last_model, flush=True)
                saver.restore(sess, self.last_model)
                print('done.', flush=True)
            elif input_model not in ['last', None]:
                print('Restoring model:\n  %s' % self.input_model, flush=True)
                saver.restore(sess, input_model)
                print('done.', flush=True)

            # summary init
            merge = tf.summary.merge_all()
              
            # train
            n_summary = 0
            for epoch in range(n_epoch):
                sess.run(last_epoch.assign(tf.add(last_epoch, 1)))

                print('Running epoch %i / %i:' % (sess.run(last_epoch)+1, sess.run(last_epoch+n_epoch-epoch)), flush=True)
                train_cost_avg = []
                train_acc_avg = []
                val_cost_avg = []
                val_acc_avg = []

                # initialize dataset iterators
                _ = sess.run([data['train']['iter_init'], data['val']['iter_init']])

                try:
                    while True:
                        self.X = data['train']['X']
                        self.y = data['train']['y']
                        _, _train_cost, _train_acc = sess.run([optimizer, train_cost, train_acc])
                        self.X = data['val']['X']
                        self.y = data['val']['y']
                        _val_cost, _val_acc = sess.run([val_cost, val_acc])
                        train_summary = sess.run(merge)
                        writer.add_summary(train_summary, n_summary)
                        writer.flush()
                        train_cost_avg.append(_train_cost)
                        train_acc_avg.append(_train_acc)
                        val_cost_avg.append(_val_cost)
                        val_acc_avg.append(_val_acc)
                        n_summary += 1
                except tf.errors.OutOfRangeError:
                    pass

                train_acc_avg = np.mean(train_acc_avg)
                val_acc_avg = np.mean(val_acc_avg)
                train_cost_avg = np.mean(train_cost_avg)
                val_cost_avg = np.mean(val_cost_avg)

                self.last_model = './workspace/%s_epoch%03i.ckpt' % (self.name, sess.run(last_epoch))
                _ = saver.save(sess, self.last_model)

                print('Epoch %i / %i completed:\n  avg. training loss: %.5e, avg. training accuracy: %.5f\n  avg. test loss: %.5e, avg. test accuracy: %.5f\n' % (sess.run(last_epoch)+1, sess.run(last_epoch+n_epoch-epoch), train_cost_avg, train_acc_avg, val_cost_avg, val_acc_avg), flush=True)
                with open(logfile, 'a') as f:
                    f.write('%i\t%.10e\t%.10e\t%.10e\t%.10e\n' % (sess.run(last_epoch)+1, train_cost_avg, train_acc_avg, val_cost_avg, val_acc_avg))