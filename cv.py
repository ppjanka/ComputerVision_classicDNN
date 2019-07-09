

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf

import GoogLeNet as GN

# 0.log
nn = GN.GoogLeNet(n_classes=2, downscale=16, name='GoogLeNet_zero')
nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.0, balance_sample=False, augment=False, pretrain_level=0)