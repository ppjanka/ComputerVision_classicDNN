

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf

import AlexNet

nn = AlexNet.AlexNet(n_classes=2, downscale=16)
nn.train(n_epoch=10)