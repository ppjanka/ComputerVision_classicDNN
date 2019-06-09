

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
        import numpy as np
        datasets = {}
        datasets['train'] = []
        datasets['val'] = []
        classes = list(os.walk(preprocessed_folder))[1:]
        no_of_classes = len(classes)
        ntotal = 0
        nclass = {}
        class2label = {}; label2class = {}; classno = 0
        img_filenames = {}
        for curr_class in classes:
            label = curr_class[0].split('/')[-1]
            img_filenames[label] = np.array([preprocessed_folder + '/' + label + '/' + x for x in curr_class[2]])
            np.random.shuffle(img_filenames[label])

            # save class identification
            nclass[label] = len(curr_class[2])
            nclass[classno] = nclass[label]
            class2label[label] = classno
            label2class[classno] = label

            classno += 1

        # balance the sample if requested
        if balance_sample:
            nmin = min(nclass.values())
            classno = 0
            for label in label2class.values():
                img_filenames[label] = img_filenames[label][:nmin]
                nclass[classno] = nmin
                nclass[label2class[classno]] = nmin
                classno += 1

        classno = 0
        for label in label2class.values():

            # calculate split for training and validation
            split_idx = int(validation_size * nclass[label])

            # save class numbers as labels
            labels = np.array(nclass[label] * [classno,])

            # create datasets
            datasets['val'].append(tf.data.Dataset.from_tensor_slices((img_filenames[label][:split_idx], labels[:split_idx])))
            datasets['train'].append(tf.data.Dataset.from_tensor_slices((img_filenames[label][split_idx:], labels[split_idx:])))

            classno += 1

        ntotal = sum(nclass.values())

        data = {}
        data['nclass'] = nclass
        data['label2class'] = label2class
        data['class2label'] = class2label

        for dataset_type in ['train', 'val']:

            data[dataset_type] = {}

            # merge all datasets
            data[dataset_type]['dataset'] = datasets[dataset_type][0]
            for dataset in datasets[dataset_type][1:]:
                data[dataset_type]['dataset'] = data[dataset_type]['dataset'].concatenate(dataset)

            # shuffle
            data[dataset_type]['dataset'] = data[dataset_type]['dataset'].shuffle(ntotal)

            # divide into batches and return iterator
            dataset = data[dataset_type]['dataset']
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            iter_next = iterator.get_next()
            # import images after drawing a batch
            img_filenames, labels = iter_next
            imgs = tf.map_fn(lambda img : tf.image.decode_jpeg(tf.read_file(img)), img_filenames, dtype=tf.uint8)
            #imgs = tf.reshape(imgs, [batch_size, *X_shape, 1]) # informs tf of the shape of the image
            imgs = tf.image.convert_image_dtype(imgs, tf.float32)
            labels = tf.one_hot(labels, no_of_classes)
            data[dataset_type]['X'] = tf.stop_gradient(imgs)
            data[dataset_type]['y'] = tf.stop_gradient(labels)
            data[dataset_type]['iter_init'] = iterator.initializer

        return data