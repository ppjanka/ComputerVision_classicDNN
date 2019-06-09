

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf

# returns tf graph handles to network training
def training_handle (y_train, nn_train, cost_function=tf.nn.softmax_cross_entropy_with_logits_v2, optimizer_function=tf.train.AdamOptimizer, optimizer_kwargs={}):
    with tf.name_scope("TRAINING") as scope:
        num_records = tf.shape(y_train)[0]
        cost = tf.reduce_mean(cost_function(labels=y_train, logits=nn_train[:num_records]), axis=0)
        tf.summary.scalar("cost_train", cost)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_train,1), tf.argmax(nn_train[:num_records],1)),tf.float32))
        tf.summary.scalar("acc_train", accuracy)
        optimizer = optimizer_function(**optimizer_kwargs).minimize(cost)
        return optimizer, cost, accuracy

# returns tf graph handles to validation data
def validation_handle (y_val, nn_val, cost_function=tf.nn.softmax_cross_entropy_with_logits_v2):
    with tf.name_scope("VALIDATION") as scope:
        num_records = tf.shape(y_val)[0]
        cost = tf.reduce_mean(cost_function(labels=y_val, logits=nn_val[:num_records]), axis=0)
        tf.summary.scalar("cost_val", cost)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_val,1), tf.argmax(nn_val[:num_records],1)), tf.float32))
        tf.summary.scalar("acc_val", accuracy)
        return cost, accuracy