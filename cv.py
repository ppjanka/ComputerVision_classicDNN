

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
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain=True, optimizer_function=tf.train.MomentumOptimizer, optimizer_kwargs={'learning_rate':0.01, 'momentum':0.9})

# 4_downscale4.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=4, name='VGGNet_zero')
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain=True, optimizer_function=tf.train.MomentumOptimizer, optimizer_kwargs={'learning_rate':0.01, 'momentum':0.9})

# 5_augmented.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=4, name='VGGNet_zero')
#nn.train(n_epoch=500, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, pretrain=True, optimizer_function=tf.train.MomentumOptimizer, optimizer_kwargs={'learning_rate':0.01, 'momentum':0.9})

# 6_downscale1.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero')
#nn.train(n_epoch=500, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain=True, pretrain_interval=500, optimizer_function=tf.train.MomentumOptimizer, optimizer_kwargs={'learning_rate':0.01, 'momentum':0.9})

# 7_stripes-dropout.log
#nn = VGGNet.VGGNet(n_classes=2, downscale=2, name='VGGNet_zero', init_stddev=0.1)
#nn.train(n_epoch=50, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0., balance_sample=False, augment=False, pretrain=True, pretrain_interval=500, optimizer_function=tf.train.GradientDescentOptimizer, optimizer_kwargs={'learning_rate':0.01})

# 8_stripes-finePretrain.log
if False:
    optimizer_function=tf.train.GradientDescentOptimizer
    optimizer_kwargs={'learning_rate':0.05}
    epochs_per_stage = 100
    start_stage = 10
    if start_stage == 0:
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=0)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0., balance_sample=False, augment=False, pretrain_level=0, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
    tf.reset_default_graph()
    current_ckpt = epochs_per_stage-1
    for pretrain_level in range(max(1,start_stage),20):
        current_ckpt = pretrain_level * epochs_per_stage - 1
        if pretrain_level == 3:
            optimizer_kwargs={'learning_rate':0.07}
        elif pretrain_level == 4:
            optimizer_kwargs={'learning_rate':0.09}
        elif pretrain_level == 5:
            optimizer_kwargs={'learning_rate':0.2}
        elif pretrain_level == 6:
            optimizer_kwargs={'learning_rate':0.35}
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=pretrain_level)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain_level=pretrain_level, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch%03i.ckpt' % current_ckpt, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
        tf.reset_default_graph()

# 9_memorize-finePretrain.log
if False:
    optimizer_function=tf.train.MomentumOptimizer
    optimizer_kwargs={'learning_rate':0.01, 'momentum':0.9}
    epochs_per_stage = 100
    start_stage = 0
    if start_stage == 0:
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=0)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain_level=0, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
    tf.reset_default_graph()
    current_ckpt = epochs_per_stage-1
    for pretrain_level in range(max(1,start_stage),20):
        current_ckpt = pretrain_level * epochs_per_stage - 1
        if False:
            if pretrain_level == 3:
                optimizer_kwargs={'learning_rate':0.07}
            elif pretrain_level == 4:
                optimizer_kwargs={'learning_rate':0.15}
            elif pretrain_level == 5:
                optimizer_kwargs={'learning_rate':0.25}
            elif pretrain_level == 6:
                optimizer_kwargs={'learning_rate':0.35}
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=pretrain_level)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain_level=pretrain_level, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch%03i.ckpt' % current_ckpt, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
        tf.reset_default_graph()

# 10_cats-vs-dogs-finePretrain.log
if False:
    optimizer_function=tf.train.MomentumOptimizer
    optimizer_kwargs={'learning_rate':0.01, 'momentum':0.9}
    epochs_per_stage = 500
    start_stage = 0
    if start_stage == 0:
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=0)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain_level=0, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
    tf.reset_default_graph()
    current_ckpt = epochs_per_stage-1
    for pretrain_level in range(max(1,start_stage),20):
        current_ckpt = pretrain_level * epochs_per_stage - 1
        if False:
            if pretrain_level == 3:
                optimizer_kwargs={'learning_rate':0.07}
            elif pretrain_level == 4:
                optimizer_kwargs={'learning_rate':0.15}
            elif pretrain_level == 5:
                optimizer_kwargs={'learning_rate':0.25}
            elif pretrain_level == 6:
                optimizer_kwargs={'learning_rate':0.35}
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=pretrain_level)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain_level=pretrain_level, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch%03i.ckpt' % current_ckpt, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
        tf.reset_default_graph()

# 11_dropout.log
if False:
    optimizer_function=tf.train.MomentumOptimizer
    optimizer_kwargs={'learning_rate':0.01, 'momentum':0.9}
    epochs_per_stage = 500
    start_stage = 0
    if start_stage == 0:
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=0)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain_level=0, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
        tf.reset_default_graph()
        current_ckpt = epochs_per_stage-1
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=0)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain_level=0, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch%03i.ckpt' % current_ckpt, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
        tf.reset_default_graph()
        current_ckpt = 2*epochs_per_stage-1
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=0)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom']}, pretrain_level=0, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch%03i.ckpt' % current_ckpt, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
        tf.reset_default_graph()
        current_ckpt = 3*epochs_per_stage-1
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=0)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate']}, pretrain_level=0, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch%03i.ckpt' % current_ckpt, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
        tf.reset_default_graph()
        current_ckpt = 4*epochs_per_stage-1
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=0)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor']}, pretrain_level=0, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch%03i.ckpt' % current_ckpt, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
        tf.reset_default_graph()
        current_ckpt = 5*epochs_per_stage-1
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=0)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor', 'color']}, pretrain_level=0, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch%03i.ckpt' % current_ckpt, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
        tf.reset_default_graph()
    current_ckpt = 6*epochs_per_stage-1
    for pretrain_level in range(max(1,start_stage),20):
        current_ckpt = (pretrain_level+5) * epochs_per_stage - 1
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=pretrain_level)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, pretrain_level=pretrain_level, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch%03i.ckpt' % current_ckpt, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
        tf.reset_default_graph()

# 12_memorizeFirst.log
if True:
    optimizer_function=tf.train.MomentumOptimizer
    optimizer_kwargs={'learning_rate':0.01, 'momentum':0.9}
    epochs_per_stage = 500
    start_stage = 14
    end_stage = 13
    if start_stage == 0:
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=0)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain_level=0, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
        tf.reset_default_graph()
    for pretrain_level in range(max(1,start_stage),end_stage):
        current_ckpt = pretrain_level * epochs_per_stage - 1
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=pretrain_level)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:1.0}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain_level=pretrain_level, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch%03i.ckpt' % current_ckpt, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
        tf.reset_default_graph()
    # add dropout and augmentation
    if False:
        current_ckpt = end_stage*epochs_per_stage-1
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=0)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=False, pretrain_level=0, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch%03i.ckpt' % current_ckpt, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
        tf.reset_default_graph()
        current_ckpt = (end_stage+1)*epochs_per_stage-1
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=0)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom']}, pretrain_level=0, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch%03i.ckpt' % current_ckpt, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
        tf.reset_default_graph()
        current_ckpt = (end_stage+2)*epochs_per_stage-1
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=0)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate']}, pretrain_level=0, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch%03i.ckpt' % current_ckpt, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
        tf.reset_default_graph()
        current_ckpt = (end_stage+3)*epochs_per_stage-1
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=0)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor']}, pretrain_level=0, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch%03i.ckpt' % current_ckpt, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
        tf.reset_default_graph()
        current_ckpt = (end_stage+4)*epochs_per_stage-1
        nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=0)
        nn.train(n_epoch=epochs_per_stage, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor', 'color']}, pretrain_level=0, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch%03i.ckpt' % current_ckpt, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)
    
    tf.reset_default_graph()

    # train some more at the end
    current_ckpt = (end_stage+5)*epochs_per_stage-1
    nn = VGGNet.VGGNet(n_classes=2, downscale=1, name='VGGNet_zero', init_stddev=0.01, pretrain_level=0)
    nn.train(n_epoch=2*epochs_per_stage, train_feed_dict={nn.dropout_keepProb:0.5}, val_feed_dict={nn.dropout_keepProb:1.0}, validation_size=0.1, balance_sample=True, augment=True, augment_dict={'todo':['zoom', 'rotate', 'flip_hor', 'color']}, pretrain_level=0, input_model='/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/workspace/VGGNet_zero_epoch%03i.ckpt' % current_ckpt, optimizer_function=optimizer_function, optimizer_kwargs=optimizer_kwargs)