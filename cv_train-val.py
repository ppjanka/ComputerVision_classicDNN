

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf

# returns tf graph handles to network training
def training_handle (X_train, y_train, nn_train, cost_function=tf.nn.softmax_cross_entropy_with_logits, optimizer_function=tf.train.AdamOptimizer, optimizer_kwargs={}):
    with tf.name_scope("TRAINING") as scope:
        cost = cost_function(labels=y_train, logits=nn_train)
        tf.summary.scalar("cost_train", cost)
        accuracy = tf.metrics.accuracy(y_train, nn_train)
        tf.summary.scalar("acc_train", accuracy)
        optimizer = optimizer_function(**optimizer_kwargs)
        return optimizer, cost, accuracy

# returns tf graph handles to validation data
def validation_handle (X_val, y_val, nn_val, cost_function=tf.nn.softmax_cross_entropy_with_logits):
    with tf.name_scope("VALIDATION") as scope:
        cost = cost_function(labels=y_val, logits=nn_val)
        tf.summary.scalar("cost_val", cost)
        accuracy = tf.metrics.accuracy(y_val, nn_val)
        tf.summary.scalar("acc_val", accuracy)
        return cost, accuracy