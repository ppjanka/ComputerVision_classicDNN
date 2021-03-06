

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf

import AlexNet

# 0.log
#nn = AlexNet.AlexNet(n_classes=2, downscale=16, name='AlexNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=False, augment=False)

# 1_balanced.log
#nn = AlexNet.AlexNet(n_classes=2, downscale=16, name='AlexNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False)

# 2_downscale4.log
#nn = AlexNet.AlexNet(n_classes=2, downscale=4, name='AlexNet_zero')
#nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False)

# 3_dropout.log
#nn = AlexNet.AlexNet(n_classes=2, downscale=4, name='AlexNet_zero')
#nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False)

# 4_downscale8.log
#nn = AlexNet.AlexNet(n_classes=2, downscale=8, name='AlexNet_zero')
#nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False)

# 5_augmented_zoom.log
#nn = AlexNet.AlexNet(n_classes=2, downscale=4, name='AlexNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom']})

# 5_augmented_zoom-rotate.log
#nn = AlexNet.AlexNet(n_classes=2, downscale=4, name='AlexNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/AlexNet_zero_epoch049.ckpt')

# 5_augmented_zoom-rotate-flip.log
#nn = AlexNet.AlexNet(n_classes=2, downscale=4, name='AlexNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/AlexNet_zero_epoch099.ckpt')

# 5_augmented_zoom-rotate-flip-color.log
#nn = AlexNet.AlexNet(n_classes=2, downscale=4, name='AlexNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor', 'color']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/AlexNet_zero_epoch149.ckpt')

# 6_downscale2.log

#nn = AlexNet.AlexNet(n_classes=2, downscale=2, name='AlexNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom']})

#nn = AlexNet.AlexNet(n_classes=2, downscale=2, name='AlexNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/AlexNet_zero_epoch049.ckpt')

#nn = AlexNet.AlexNet(n_classes=2, downscale=2, name='AlexNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/AlexNet_zero_epoch099.ckpt')

#nn = AlexNet.AlexNet(n_classes=2, downscale=2, name='AlexNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor', 'color']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/AlexNet_zero_epoch149.ckpt')

# 7_downscale1.log

#nn = AlexNet.AlexNet(n_classes=2, downscale=1, name='AlexNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom']})

#nn = AlexNet.AlexNet(n_classes=2, downscale=1, name='AlexNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/AlexNet_zero_epoch049.ckpt')

#nn = AlexNet.AlexNet(n_classes=2, downscale=1, name='AlexNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/AlexNet_zero_epoch099.ckpt')

#nn = AlexNet.AlexNet(n_classes=2, downscale=1, name='AlexNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor', 'color']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/AlexNet_zero_epoch149.ckpt')

# 8_downscale1-long.log

nn = AlexNet.AlexNet(n_classes=2, downscale=1, name='AlexNet_zero')
nn.train(n_epoch=100, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor', 'color']}, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/AlexNet_zero_epoch199.ckpt')