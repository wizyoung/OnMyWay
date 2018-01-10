# coding: utf-8
import os
import tensorflow as tf
import numpy as np
from cytf.arg_scope import *
from cytf.layer import conv2d, dense, flatten, max_pool, activation, dropout
from cytf.initializers import he_normal

LOG_DIR = './log/'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1], name='input_image')
    tf.summary.image('input_image', x_image, 9)
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='input_y')
    train_flag = tf.placeholder(tf.bool)
    dropout_param = tf.placeholder(tf.float32)

with arg_scope([conv2d], padding='same', initializer='he_normal'):
    with arg_scope([max_pool], ksize=[2, 2], stride=2, padding='valid'):
        net = conv2d(x_image, 32, [5, 5], 1,
                 name='layer1/conv2', activation='relu',
                 BN=True, regualizer={'l2': 4e-4}, train=train_flag)
        net = max_pool(net, name='layer1/maxpool')

        net = conv2d(net, 64, [5, 5], 1, 
                     name='layer2/conv2', activation='relu',
                     BN=True, regualizer={'l2': 4e-4}, train=train_flag)
        net = max_pool(net, name='layer2/maxpool')  

        net = conv2d(net, 128, [3, 3], 1, 
                     name='layer3/conv2', activation='relu',
                     BN=True, regualizer={'l2': 4e-4}, train=train_flag)
        net = max_pool(net, name='layer3/maxpool') 

        net = conv2d(net, 128, [1, 1], 1, 
                     name='layer4/conv3_1X1', activation='relu', train=train_flag)

        flattened = flatten(net, name='flatten')
        net = dense(flattened, 1024, name='layer5/dense', initializer='he_normal', BN=True, 
                regualizer={'l2': 4e-4}, activation='relu', train=train_flag)
        net = dropout(net, keep_prob=dropout_param, name='layer5/dropout')
        y_pred = dense(net, 10, name='layer6/dense', initializer='he_normal', BN=True, 
                regualizer={'l2': 4e-4}, train=train_flag)
        net = dropout(net, keep_prob=dropout_param, name='layer6/dropout')

with tf.Session() as sess:
    with tf.name_scope('loss_function'):
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_))
        tf.add_to_collection('losses', cross_entropy)
        losses = tf.add_n(tf.get_collection('losses'), name='total_loss')
        
    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(losses)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))

        # why this fails?
        # def acc_train():
        #     acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='train_acc')
        #     with tf.name_scope('acc_summary'):
        #         tf.summary.scalar('acc_train', acc)
        #     return acc

        # def acc_test():
        #     acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='acc_test')  
        #     with tf.name_scope('acc_summary'):
        #         tf.summary.scalar('acc_train', acc)
        #     return acc

        # accuracy = tf.cond(tf.equal(train_flag, tf.constant(True)), acc_train, acc_test)

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope('acc_summary'):
        tf.summary.scalar('accuracy', accuracy)

    # misc init
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    for i in range(20000):
        print i
        batch = mnist.train.next_batch(50)
        summary, _ = sess.run([merged, train_step],
                              feed_dict={x: batch[0], y_: batch[1], train_flag: True, dropout_param: 0.5})
        writer.add_summary(summary, i)

        if i % 100 == 10:
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, train_flag: False, dropout_param: 1.0})
            print test_acc
    
    test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, train_flag: False, dropout_param: 1.0})
    print test_acc

    writer.close()

