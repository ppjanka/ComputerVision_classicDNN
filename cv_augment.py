

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf
import numpy as np

# image augmentation routine, based on: https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
def augment_handle (X, zoom_min=0.6, zoom_max=1.4, brightness_range=0.05, contrast_min=0.7, contrast_max=1.3):
    
    n_images = tf.shape(X)[0]
    image_shape = tf.shape(X)[1:3]
    # random rotate
    X = tf.image.rotate(X, tf.random.uniform([n_images,], minval=0., maxval=2.*np.pi))
    # random flip
    choice = tf.random.uniform([n_images,], minval=0., maxval=1.)
    X = tf.cond(choice < 0.5, lambda : X, lambda : tf.image.random_flip_left_right(X))
    choice = tf.random.uniform([n_images,], minval=0., maxval=1.)
    X = tf.cond(choice < 0.5, lambda : X, lambda : tf.image.random_flip_up_down(X))
    # random zoom
    zoom = tf.random.uniform([n_images,], minval=zoom_min, maxval=zoom_max)
    X = tf.image.resize_images(X, size=(zoom*image_shape), preserve_aspect_ratio=True)
    X = tf.image.resize_image_with_crop_or_pad(X, image_shape)
    # random color
    X = tf.image.random_brightness(X, brightness_range)
    X = tf.image.random_contrast(X, contrast_min, contrast_max)

    return X