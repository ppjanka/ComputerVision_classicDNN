

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf

def get_filenames (label, ext, preprocessed=False):
    import os
    import glob
    import numpy as np
    base_dir = os.getcwd()
    if not preprocessed:
        stem = base_dir + '/data/' + label + '/'
    else:
        stem = base_dir + '/data_preprocessed/' + label + '/'
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

# returns a handle to iterator.get_next() of images Dataset to be connected to a comp graph
def image_batch_handle (preprocessed_folder):

    with tf.name_scope('IMG_INPUT'):

        import os
        datasets = []
        classes = list(os.walk(preprocessed_folder))[1:]
        ntotal = 0
        nclass = {}
        class2label = {}; label2class = {}; classno = 0
        for curr_class in classes:
            label = curr_class[0].split('/')[-1]
            img_filenames = preprocessed_folder + '/' + label + '/' + tf.constant(curr_class[2])

            # save class identification
            nclass[label] = len(curr_class[2])
            nclass[classno] = nclass[label]
            class2label[label] = classno
            label2class[classno] = label

            # save class numbers as labels
            labels = tf.constant(nclass[label] * [classno,])

            # create a dataset
            datasets.append(tf.data.Dataset.from_tensor_slices((img_filenames, labels)))

            classno += 1

        ntotal = sum(nclass.values())

        # merge all datasets
        final_dataset = datasets[0]
        for dataset in datasets[1:]:
            final_dataset = final_dataset.concatenate(dataset)

        # shuffle and return iterator
        final_dataset = final_dataset.shuffle(ntotal)
        final_dataset = final_dataset.batch(batch_size)
        iterator = final_dataset.make_one_shot_iterator()
        iter_next = iterator.get_next()

        # import images after drawing a batch
        img_filenames, labels = iter_next
        imgs = tf.map_fn(lambda img : tf.image.decode_jpeg(tf.read_file(img)), img_filenames, dtype=tf.uint8)
        imgs = tf.image.convert_image_dtype(imgs, tf.float32)
        labels = tf.one_hot(labels, classno)

        return imgs, labels, nclass, class2label, label2class