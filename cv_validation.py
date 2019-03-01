

exec(open("./globals.py").read()) # read global variables

import tensorflow as tf

# returns tf graph handles to validation data
def validation_handle (X_val, y_val, nn_val, cost_function=tf.nn.softmax_cross_entropy_with_logits):
    with tf.name_scope("VALIDATION") as scope:
        cost = cost_function(labels=y_val, logits=nn_val)
        tf.summary.scalar("cost_val", cost)
        accuracy = tf.metrics.accuracy(y_val, nn_val)
        tf.summary.scalar("acc_val", accuracy)
        return cost, accuracy