

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
nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero')
nn.train(n_epoch=500, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain=True, pretrain_interval=500, optimizer=tf.train.MomentumOptimizer, optimizer_kwargs={'learning_rate':0.01, 'momentum':0.9})





# 2_downscale4.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=4, name='VGGNet_zero')
#nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False)

# 3_dropout.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=4, name='VGGNet_zero')
#nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False)

# 4_downscale8.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=8, name='VGGNet_zero')
#nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False)

# 5_augmented_zoom.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=4, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom']})

# 5_augmented_zoom-rotate.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=4, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch049.ckpt')

# 5_augmented_zoom-rotate-flip.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=4, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch099.ckpt')

# 5_augmented_zoom-rotate-flip-color.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=4, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor', 'color']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch149.ckpt')

# 6_downscale2.log

#nn = VGGNet.VGGNet(n_classes=2, downscale=2, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom']})

#nn = VGGNet.VGGNet(n_classes=2, downscale=2, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch049.ckpt')

#nn = VGGNet.VGGNet(n_classes=2, downscale=2, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch099.ckpt')

#nn = VGGNet.VGGNet(n_classes=2, downscale=2, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor', 'color']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch149.ckpt')

# 7_downscale1.log

#nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom']})

#nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch049.ckpt')

#nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch099.ckpt')

#nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor', 'color']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch149.ckpt')

# 8_downscale1-long.log

#nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero')
#nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor', 'color']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch199.ckpt')