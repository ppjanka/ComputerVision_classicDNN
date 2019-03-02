
# [cv.py] Image recognition NN training and testing

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf

import cv_io as cvio

if __name__ == '__main__':

    preprocessed_folder = './data_preprocessed'

    # input
    import os
    if not os.direxists(preprocessed_folder):
        print(preprocessed_folder + ' does not exist. Please preprocess data using preprocess.py first.')
        return 0
    data = cvio.image_batch_handle(preprocessed_folder)
