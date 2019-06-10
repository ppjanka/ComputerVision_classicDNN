

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf

import ZFNet

# 0.log
#nn = ZFNet.ZFNet(n_classes=2, downscale=16, name='ZFNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=False, augment=False)

# 1_balanced.log
#nn = ZFNet.ZFNet(n_classes=2, downscale=16, name='ZFNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False)

# 2_balanced-long.log
#nn = ZFNet.ZFNet(n_classes=2, downscale=16, name='ZFNet_zero')
#nn.train(n_epoch=30, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/NNs/ZFNet_logs/1_balanced.ckpt/ZFNet_zero_epoch049.ckpt')

# 3_dropout.log
#nn = ZFNet.ZFNet(n_classes=2, downscale=16, name='ZFNet_zero')
#nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False)

# 4_augmented.log

#nn = ZFNet.ZFNet(n_classes=2, downscale=1, name='ZFNet_zero')
#nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom']})

#nn = ZFNet.ZFNet(n_classes=2, downscale=1, name='ZFNet_zero')
#nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/ZFNet_zero_epoch099.ckpt')

#nn = ZFNet.ZFNet(n_classes=2, downscale=1, name='ZFNet_zero')
#nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/ZFNet_zero_epoch199.ckpt')

nn = ZFNet.ZFNet(n_classes=2, downscale=1, name='ZFNet_zero')
nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor', 'color']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/ZFNet_zero_epoch299.ckpt')