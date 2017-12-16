# coding: utf-8
import os
import tensorflow as tf
import numpy as np
from cytf.layer import conv2d, dense, flatten, max_pool, activation
from cytf.initializers import he_normal
from tensorflow.contrib.tensorboard.plugins import projector

LOG_DIR = './log/'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1], name='input_image')
    tf.summary.image('input_image', x_image, 3)
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='input_y')

net = conv2d(x_image, 32, [5, 5], 1, 'same', initializer='he_normal',
             name='layer1/conv2', TFboard_recording=[1, 1, 1, 1, 0],
             BN=True, regualizer={'l2': 4e-4})
net = activation(net, 'relu', name='layer1/relu', TFboard_recording=True)
net = max_pool(net, [2, 2], 2, 'valid', name='layer1/maxpool')
net = conv2d(net, 64, [5, 5], 1, 'same', initializer='he_normal',
             name='layer2/conv2', activation='relu', TFboard_recording=[1, 1, 1, 1, 1],
             BN=True, regualizer={'l2': 4e-4})
net = max_pool(net, [2, 2], 2, 'valid', name='layer2/maxpool')   
flattened = flatten(net, name='flatten')
net = dense(flattened, 1024, name='layer3/dense', initializer='he_normal', BN=True, 
            regualizer={'l2': 4e-4}, TFboard_recording=[1, 1, 1, 1, 1], activation='relu')
y_pred = dense(net, 10, name='layer4/dense', initializer='he_normal', BN=True, 
            regualizer={'l2': 4e-4}, TFboard_recording=[1, 1, 1, 1, 0])

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
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope('acc_summary'):
        tf.summary.scalar('accuracy', accuracy)
    
    embedding_input = pred
    embedding_size = embedding_input.get_shape().as_list()[-1]

    # tut
    merged = tf.summary.merge_all()

    embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
    assignment = embedding.assign(embedding_input)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    config = projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = LOG_DIR + 'img.png'
    embedding_config.metadata_path = LOG_DIR + 'l.tsv'
    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(writer, config)

    # def TFboard_embedding(num_input, input, summary_write):
    #     '''
    #     '''
    #     with tf.variable_scope('TFboard_Embedding'):
    #         # calc high_dim
    #         embedding_var = tf.Variable(tf.zeros([num_input, high_dim]), name='embedding_var')
    #         # ass

    #         config = tensorflow.contrib.tensorboard.plugins.projector.ProjectorConfig()
    #         #TODO Q: can add more? 
    #         embedding_config = config.embeddings.add()
    #         embedding_config.tensor_name = 



    for i in range(20000):
        batch = mnist.train.next_batch(50)
        summary, _ = sess.run([merged, train_step],
                              feed_dict={x: batch[0], y_: batch[1]})
        writer.add_summary(summary, i)

        if i % 10 == 0:
            sess.run(assignment, feed_dict={x: mnist.test.images[:1024], y_: mnist.test.labels[:1024]})
            saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'), i)
            print 'step:', i 
    writer.close()

