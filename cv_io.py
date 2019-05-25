

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
def image_batch_handle (preprocessed_folder, validation_size=0.2, balance_sample=True):

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

        # balance the sample if requested
        if balance_sample:
            nmin = min(nclass.values())
            for classno in range(len(datasets)):
                datasets[classno].shuffle(nclass[classno])
                datasets[classno].take(nmin)
                nclass[classno] = nmin
                nclass[label2class[classno]] = nmin

        # merge all datasets
        final_dataset = datasets[0]
        for dataset in datasets[1:]:
            final_dataset = final_dataset.concatenate(dataset)

        data = {}
        data['nclass'] = nclass
        data['label2class'] = label2class
        data['class2label'] = class2label

        # shuffle
        final_dataset = final_dataset.shuffle(ntotal)

        # split into training and validation datasets
        data['train'] = {}; data['val'] = {}
        int_validation_size = int(validation_size * ntotal)
        data['train']['dataset'] = final_dataset.skip(int_validation_size)
        data['val']['dataset'] = final_dataset.take(int_validation_size)

        for dataset_type in ['train', 'val']:
            # divide into batches and return iterator
            dataset = data[dataset_type]['dataset']
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            iter_next = iterator.get_next()
            # import images after drawing a batch
            img_filenames, labels = iter_next
            imgs = tf.map_fn(lambda img : tf.image.decode_jpeg(tf.read_file(img)), img_filenames, dtype=tf.uint8)
            imgs = tf.image.convert_image_dtype(imgs, tf.float32)
            labels = tf.one_hot(labels, classno)
            data[dataset_type]['X'] = tf.stop_gradient(imgs)
            data[dataset_type]['y'] = tf.stop_gradient(labels)
            data[dataset_type]['iter_init'] = iterator.initializer

        return data