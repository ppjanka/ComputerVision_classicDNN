

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf
import numpy as np

# image augmentation routine, based on: https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
def augment_handle (X, zoom=0.9, brightness_range=0.05, contrast_min=0.9, contrast_max=1.1, angle_min=-0.25*np.pi, angle_max=0.25*np.pi, todo=['rotate', 'flip_hor', 'flip_vert', 'zoom', 'color']):
    
    n_images = tf.shape(X)[0]
    image_shape = tf.shape(X)[1:3]
    if 'rotate' in todo:
        # random rotate
        X = tf.contrib.image.rotate(X, tf.random.uniform([n_images,], minval=angle_min, maxval=angle_max))
    if 'flip_hor' in todo:
        # random flip
        choice = tf.random.uniform([n_images,], minval=0., maxval=1.)
        X = tf.where(choice < 0.5, X, tf.image.random_flip_left_right(X))
    if 'flip_vert' in todo:
        choice = tf.random.uniform([n_images,], minval=0., maxval=1.)
        X = tf.where(choice < 0.5, X, tf.image.random_flip_up_down(X))
    if 'zoom' in todo:
        # random zoom
        X = tf.image.random_crop(X, tf.concat([n_images * tf.ones(1, tf.int32), tf.cast(zoom*tf.cast(image_shape, tf.float32), tf.int32), tf.ones(1, tf.int32)], axis=0))
        X = tf.image.resize_image_with_crop_or_pad(X, image_shape[0], image_shape[1])
    if 'color' in todo:
        # random color
        X = tf.image.random_brightness(X, brightness_range)
        X = tf.image.random_contrast(X, contrast_min, contrast_max)

    return X