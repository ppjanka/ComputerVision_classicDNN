# ComputerVision_classicDNN
Playing with classical Computer Vision DNNs from the literature.

**For my walk-throughs trying out different architectures go directly to NNs/*.ipynb.**

1) Data and preprocessing:
 - data_black-vs-white: contains two jpg files with black and white images for sanity checks
 - data_cats-vs-dogs/*.txt: list of images (of cats and dogs) from ImageNet (http://image-net.org/) to be used as training/validation dataset
 - data_cats-vs-dogs/download.py: a script for downloading images from *.txt files
 - preprocess.py: a script used to load, transform to grayscale, reshape, pad images downloaded with data/download.py, as well as filter out instances of 'image not found' downloads; the images are then saved to data_preprocessed folder

2) "Scaffolding":
 - globals.py: some global definitions
 - cv_io.py: defines tensorflow handles of the data input
 - cv_augment.py: adds data augmentation to the computational graph
 - cv_train_val.py: training and validation sections of the comp. graph
 - cv.py: the main driving script
 - cv_vis.py: convenient tensorboard-like plotting (using Plotly, https://plot.ly/) to display training statistics in a Jupyter Notebook
 
3) The networks:
 - NNs/NN_parent: the parent class for all neural networks; handles basic tasks, including training
 - AlexNet (Krizhevsky, Sutskever, and Hinton, 2012):
   - NNs/AlexNet.pdf: the original paper, from http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
   - NNs/AlexNet.py: defines AlexNet and its properties
   - NNs/AlexNet.ipynb: my run through AlexNet (uses: NNs/AlexNet_logs, the cv.py used is contained in AlexNet_run.py)
