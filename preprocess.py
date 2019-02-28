
# TODO: it would be much more efficient to process large batches of files (to prevent waiting for IO to/from the GPU) -- that requires error handling for corrupt files within the TF graph itself

exec(open("./globals.py").read()) # read global variables

batch_size = 1
output_pathstem = '/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/data_preprocessed/'
extensions = ['jpg', 'png', 'bmp', 'gif'] # raw input file formats

# sources:
# - https://www.tensorflow.org/guide/datasets

import tensorflow as tf

def _preprocess_image (filename, ext):
    image_string = tf.read_file(filename[0])
    if ext == 'jpg':
        image_decoded = tf.image.decode_jpeg(image_string)
    elif ext == 'png':
        image_decoded = tf.image.decode_png(image_string, channels=3)
    elif ext == 'bmp':
        image_decoded = tf.image.decode_bmp(image_string)
    elif ext == 'gif':
        image_decoded = tf.image.decode_gif(image_string)
        image_decoded = tf.squeeze(image_decoded, axis=0)
    else:
        return tf.random.normal([*X_shape, 1])
    # counting unique color values will detect and allow to discard text etc
    no_unique = tf.size(tf.unique(tf.reshape(image_decoded, [-1,]))[0])
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    # calculate standard deviation to decide if calculate further
    mean, var = tf.nn.moments(image_decoded, axes=[0,1,2])
    image_grayscale = tf.cond(tf.shape(image_decoded)[2] > 1, \
        lambda : tf.image.rgb_to_grayscale(image_decoded), \
        lambda : image_decoded)
    image_resized = tf.image.resize_images(image_grayscale, size=X_shape, preserve_aspect_ratio=True)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_resized, *X_shape) # will pad with 0s
    image_standard = tf.image.convert_image_dtype(image_resized, tf.uint8)
    image_encoded = tf.image.encode_jpeg(image_standard)
    # apply stddev condition
    image_result = tf.cond( \
        tf.logical_and(var > min_variance, no_unique > min_unique), \
        lambda : image_encoded, \
        lambda : tf.constant('', dtype=tf.string)) # don't do other steps if we filter
    return var, no_unique, image_result, filename[1]

preprocess_image = {}
for extension in extensions:
    preprocess_image[extension] = (lambda filename : _preprocess_image(filename, extension))

def get_filenames (label, ext):
    import os
    import glob
    import numpy as np
    base_dir = os.getcwd()
    stem = base_dir + '/data/' + label + '/'
    # return the list of filepaths for the given extension
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    # rename files in the data folder to comply with Unicode-8
    filenames = glob.glob(stem + '*.' + ''.join(list(map(either,ext))))
    if np.array([not x.split('/')[-1].split('.')[0].isdigit() for x in filenames]).any():
        print('Renaming files for label %s... ' % label, end='', flush=True)
        i = 0
        for filename in filenames:
            os.rename(filename, stem + ('%05i.' % i) + filename.split('/')[-1].split('.')[1])
            i += 1
        print('done.', flush=True)
    return glob.glob(stem + '*.' + ''.join(list(map(either,ext))))

if __name__ == '__main__':

    processed = 0
    io_errors = 0
    for label in ['cats','dogs']:
        for extension in extensions:
            filenames = [[],[]]
            filenames[0] = get_filenames(label, extension)
            filenames[1] = [output_pathstem + label + '/' + ('%05i' % i) + '.jpg' for i in range(len(filenames[0]))]
            if len(filenames) < 1 : continue
            filenames = tf.transpose(tf.constant(filenames, dtype=tf.string))

            dataset = tf.data.Dataset.from_tensor_slices((filenames))
            dataset = dataset.map(preprocess_image[extension]).batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()

            var, no_unique, image_processed, filename = next_element
            condition = tf.logical_and(var > min_variance, no_unique > min_unique)
            image_processed = tf.boolean_mask(image_processed, condition)
            filename = tf.boolean_mask(filename, condition)

            with tf.Session() as sess:
                try:
                    while True:
                        try:
                            img, fn = sess.run([image_processed, filename])
                            # write file without GPU IO, using cpu
                            with tf.device('/device:CPU:0'):
                                for i in range(len(fn)):
                                    _ = sess.run(tf.write_file(fn[i], img[i]))
                            processed += 1
                        except tf.errors.InvalidArgumentError:
                            io_errors += 1
                        except tf.errors.DataLossError:
                            io_errors += 1
                except tf.errors.OutOfRangeError:
                    pass
            #from tensorflow.python.client import device_lib
            #print(device_lib.list_local_devices())

    print('Processed %i files. %i aborted due to IO errors.' % (processed, io_errors))