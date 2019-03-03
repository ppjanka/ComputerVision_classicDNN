

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf

import cv_io as cvio
import cv_train_val as cvtrain

class NN:

    def __init__ (self, n_classes, img_size=X_shape, tensorboard_verbose=3, name='NN', init_model=None):

        self.name = name
        self.last_model = init_model

        self.X = tf.placeholder(tf.float32, [None, *img_size, 1], name='INPUT')
        self.y = tf.placeholder(tf.float32, [None, n_classes, 1], name='LABELS')

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

    def train (self, preprocessed_folder='./data_preprocessed', validation_size=0.2, n_epoch=1, input_model='last', output_model=None):

        # initialize
        tf.logging.set_verbosity(tf.logging.INFO)
        last_epoch = tf.Variable(-1, name='last_epoch')
        saver = tf.train.Saver()

        # input
        import os
        if not os.direxists(preprocessed_folder):
            print(preprocessed_folder + ' does not exist. Please preprocess data using preprocess.py first.')
            return 0
        data = cvio.image_batch_handle(preprocessed_folder, validation_size=validation_size)

        # setup training and validation handles
        optimizer, train_cost, train_acc = cvtrain.training_handle (data['train']['X'], data['train']['y'], self.nn)
        val_cost, val_acc = cvtrain.validation_handle(data['val']['X'], data['val']['y'], self.nn)

        with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
              
            writer = tf.summary.FileWriter('./workspace/log_%s/1/train' % self.name, sess.graph)

            # restore model or initialize variables
            if input_model == 'last' and self.last_model != None:
                print('Restoring the last available model:\n  %s' % self.last_model, flush=True)
                saver.restore(sess, self.last_model)
                print('done.', flush=True)
            elif input_model not in ['last', None]:
                print('Restoring model:\n  %s' % self.input_model, flush=True)
                saver.restore(sess, input_model)
                print('done.', flush=True)
            else:
                print('Initializing all variables.. ', end='', flush=True)
                sess.run(tf.global_variables_initializer())
                print('done.', flush=True)

            # save initial state
            merge = tf.summary.merge_all()
            train_summary = sess.run(merge)
            writer.add_summary(train_summary)
            train_writer.flush()
              
            # train
            for epoch in range(n_epoch):
                sess.run(last_epoch.assign(tf.add(last_epoch, 1)))

                print('Running epoch %i / %i:' % (sess.run(last_epoch)+1, sess.run(last_epoch+n_epoch-epoch)), flush=True)
                train_cost_avg = []
                train_acc_avg = []
                val_cost_avg = []
                val_acc_avg = []

                try:
                    while True:
                        merge = tf.summary.merge_all()
                        _, _train_cost, _train_acc = sess.run([optimizer, train_cost, train_acc])
                        _val_cost, _val_acc = sess.run([val_cost, val_acc])
                        train_summary = sess.run(merge)
                        writer.add_summary(train_summary)
                        train_writer.flush()
                        train_cost_avg.append(_train_cost)
                        train_acc_avg.append(_train_acc)
                        val_cost_avg.append(_val_cost)
                        val_acc_avg.append(_val_acc)
                    except tf.errors.OutOfRangeError:
                        pass

                train_acc_avg = np.mean(train_acc_avg)
                val_acc_avg = np.mean(val_acc_avg)
                train_cost_avg = np.mean(train_cost_avg)
                val_cost_avg = np.mean(val_cost_avg)

                self.last_model = './workspace/%s_epoch%03i.ckpt' % (self.name, sess.run(last_epoch))
                _ = saver.save(sess, self.last_model)

                print('Epoch %i / %i completed:\n  avg. training loss: %.5e, avg. training accuracy: %.5f\n  avg. test loss: %.5e, avg. test accuracy: %.5f\n' % (sess.run(last_epoch)+1, sess.run(last_epoch+n_epoch-epoch), train_cost_avg, train_acc_avg, test_cost_avg, test_acc_avg), flush=True)