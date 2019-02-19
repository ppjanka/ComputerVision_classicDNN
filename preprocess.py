
batch_size = 1
output_pathstem = '/DATA/Dropbox/LOOTRPV/Personal_programming/MachineLearning/Tutorials/ImageRecognition/data_preprocessed/'

goal_shape = [256,256]
extensions = ['jpg', 'png', 'bmp', 'gif']
output_color_depth = 128 # output image will be between 0 and 2x this value

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
        return tf.random.normal([*goal_shape, 1])
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_grayscale = tf.cond(tf.shape(image_decoded)[2] > 1, \
        lambda: tf.image.rgb_to_grayscale(image_decoded), \
        lambda : image_decoded)
    image_resized = tf.image.resize_images(image_grayscale, size=goal_shape, preserve_aspect_ratio=True)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_resized, *goal_shape) # will pad with 0s
    image_standard = image_resized
    #image_standard = tf.image.per_image_standardization(image_resized) * output_color_depth + output_color_depth
    image_standard = tf.image.convert_image_dtype(image_standard, tf.uint8)
    image_encoded = tf.image.encode_jpeg(image_standard)
    return image_encoded, filename[1]

preprocess_image = {}
for extension in extensions:
    preprocess_image[extension] = (lambda filename : _preprocess_image(filename, extension))

def get_filenames (label, ext):
    import os
    import glob
    base_dir = os.getcwd()
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob.glob(base_dir + '/data/' + label + '/*.' + ''.join(list(map(either,ext))))

if __name__ == '__main__':

    processed = 0
    io_errors = 0
    for label in ['cats','dogs']:
        for extension in extensions:
            filenames = [[],[]]
            filenames[0] = get_filenames(label, extension)[:2]
            filenames[1] = [output_pathstem + label + '/' + str(i) + '.jpg' for i in range(len(filenames[0]))]
            if len(filenames) < 1 : continue
            filenames = tf.transpose(tf.constant(filenames, dtype=tf.string))

            dataset = tf.data.Dataset.from_tensor_slices((filenames))
            dataset = dataset.map(preprocess_image[extension]).batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()

            with tf.Session() as sess:
                try:
                    while True:
                        try:
                            image_processed, filename = sess.run(next_element)
                            sess.run(tf.write_file(filename[0], image_processed[0]))
                            processed += 1
                        except tf.errors.InvalidArgumentError:
                            io_errors += 1
                except tf.errors.OutOfRangeError:
                    pass

    print('Processed %i files. %i aborted due to IO errors.' % (processed, io_errors))