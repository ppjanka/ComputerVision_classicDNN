# ComputerVision_classicDNN
Playing with classical Computer Vision DNNs from the literature.

File description:
 - data/*.txt: list of images (of cats and dogs) from ImageNet (http://image-net.org/) to be used as training/validation dataset
 - data/download.py: a script to download images from *.txt files
 - preprocess.py: a script to load, transform to grayscale, reshape, pad images downloaded with data/download.py, as well as filter out instances of 'image not found' downloads; the images are then saved to data_preprocessed folder
