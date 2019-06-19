

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf

import VGGNet

# 0.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=16, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False)

# 1_downscale8.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=8, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False)

# 2_pretrainA.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=8, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain=True)

# 3_momentum.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=8, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain=True, optimizer=tf.train.MomentumOptimizer, optimizer_kwargs={'learning_rate':0.01, 'momentum':0.9})

# 4_downscale4.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=4, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain=True, optimizer=tf.train.MomentumOptimizer, optimizer_kwargs={'learning_rate':0.01, 'momentum':0.9})

# 5_augmented.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=4, name='VGGNet_zero')
#nn.train(n_epoch=500, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, pretrain=True, optimizer=tf.train.MomentumOptimizer, optimizer_kwargs={'learning_rate':0.01, 'momentum':0.9})

# 6_downscale1.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero')
#nn.train(n_epoch=500, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain=True, pretrain_interval=500, optimizer=tf.train.MomentumOptimizer, optimizer_kwargs={'learning_rate':0.01, 'momentum':0.9})

# 7_stripes-dropout.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=2, name='VGGNet_zero', init_stddev=0.1)
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0., balance_sample=False, augment=False, pretrain=True, pretrain_interval=500, optimizer=tf.train.GradientDescentOptimizer, optimizer_kwargs={'learning_rate':0.01})

# 8_stripes-finePretrain.log
nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01)
nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0., balance_sample=True, augment=False, pretrain_level=0)
tf.reset_default_graph()
nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01)
nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0., balance_sample=True, augment=False, pretrain_level=1, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch099.ckpt')
tf.reset_default_graph()
nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01)
nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0., balance_sample=True, augment=False, pretrain_level=2, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch199.ckpt')
tf.reset_default_graph()
nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01)
nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0., balance_sample=True, augment=False, pretrain_level=3, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch299.ckpt')