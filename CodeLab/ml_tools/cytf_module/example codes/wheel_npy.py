#coding: utf-8
import numpy as np
import tensorflow as tf
from cytf.arg_scope import *
from cytf.layer import conv2d, dense, flatten, max_pool, activation, dropout
from cytf.initializers import he_normal

train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')

def to_categorical(y, nb_classes=None):
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

train_label = np.zeros(train_data.shape[0], dtype=int)
train_label[600:] = 1
train_label = to_categorical(train_label, 2)
test_label = np.zeros(test_data.shape[0], dtype=int)
test_label[300:] = 1
test_label = to_categorical(test_label, 2)

print 'train data shape:{}, test data shape:{}'.format(train_data.shape, test_data.shape)

def model(x, train_flag, dropout_param):
    with arg_scope([conv2d], padding='same', kernel_size=[3, 3], stride=[1, 1], initializer='he_normal', activation='relu', train=train_flag):
        with arg_scope([max_pool], ksize=[2, 2], stride=2, padding='valid'):
            with arg_scope([dense], initializer='he_normal', BN=True, train=train_flag, regualizer={'l2': 4e-4}):
                net = conv2d(x, 64, BN=True, regualizer={'l2': 4e-4}, name='layer1/conv2_1')
                net = conv2d(net, 64, BN=True, regualizer={'l2': 4e-4}, name='layer1/conv2_2')
                net = max_pool(net, name='layer1/maxpool')

                net = conv2d(net, 128, BN=True, regualizer={'l2': 4e-4}, name='layer2/conv2_1')
                net = conv2d(net, 128, BN=True, regualizer={'l2': 4e-4}, name='layer2/conv2_2')
                net = max_pool(net, name='layer2/maxpool')

                net = conv2d(net, 256, BN=True, regualizer={'l2': 4e-4}, name='layer3/conv2_1')
                net = conv2d(net, 256, BN=True, regualizer={'l2': 4e-4}, name='layer3/conv2_2')
                net = conv2d(net, 256, kernel_size=[1, 1], BN=True, regualizer={'l2': 4e-4}, name='layer3/conv2_3')
                net = max_pool(net, name='layer3/maxpool')

                net = conv2d(net, 512, BN=True, regualizer={'l2': 4e-4}, name='layer4/conv2_1')
                net = conv2d(net, 512, BN=True, regualizer={'l2': 4e-4}, name='layer4/conv2_2')
                net = max_pool(net, name='layer4/maxpool')

                flattened = flatten(net, name='flatten')

                net = dense(flattened, 1024, activation='relu', name='layer5/dense')
                net = dropout(flattened, keep_prob=dropout_param, name='layer5/dropout')

                net = dense(net, 1024, activation='relu', name='layer6/dense')
                net = dropout(net, keep_prob=dropout_param, name='layer6/dropout')

                y_pred = dense(net, 2, name='layer7/out')

                return y_pred

def train_process(model, train, input_x, y_true, test_data, test_label, lr=1e-4, dropout=0.5, batch_size=50):
    with tf.variable_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 112, 112, 3], name='input_image')
        y = tf.placeholder(tf.float32, shape=[None, 2], name='input_y')
        train_flag = tf.placeholder(tf.bool)
        dropout_param = tf.placeholder(tf.float32)

    y_pred = model(x, train_flag, dropout_param=dropout_param)

    with tf.name_scope('loss_function'):
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
        tf.add_to_collection('losses', cross_entropy)
        losses = tf.add_n(tf.get_collection('losses'), name='total_loss')

    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(lr).minimize(losses)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='acc')
    
    with tf.name_scope('acc_summary'):
        tf.summary.scalar('accuracy', acc)

    with tf.Session() as sess:
        # misc init
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./tfboard', sess.graph)
        saver = tf.train.Saver()

        for epoch in range(300):
            batch_num = int(np.ceil(input_x.shape[0] / float(batch_size)))
            index_array = np.arange(input_x.shape[0])
            np.random.shuffle(index_array)

            for idx in range(batch_num):
                print 'epoch:{}, index:{}/{}'.format(epoch, idx, batch_num)
                index_arange = list(index_array[idx * batch_size: min((idx + 1) * batch_size, input_x.shape[0])])
                data_in = input_x[index_arange]
                label_in = y_true[index_arange]
                
                summary, _ = sess.run([merged, train_step], feed_dict={x: data_in, y: label_in, train_flag: train, dropout_param: dropout})
                writer.add_summary(summary, idx)

            #test
            batch_num_test = int(np.ceil(test_data.shape[0] / float(batch_size)))
            test_acc = 0
            for idx in range(batch_num_test):
                test_x = test_data[idx * batch_size: min((idx + 1) * batch_size, test_data.shape[0])]
                test_y = test_label[idx * batch_size: min((idx + 1) * batch_size, test_data.shape[0])]

                test_acc += sess.run(tf.reduce_sum(tf.cast(correct_prediction, tf.float32)), 
                                    feed_dict={x: test_x, y: test_y, train_flag: False, dropout_param: 1.0})
                                
            test_acc = test_acc / test_data.shape[0]
            # test_acc = sess.run(acc, feed_dict={x: test_data, y: test_label, train_flag: False, dropout_param: 1.0})
            print epoch, test_acc
            saver.save(sess, 'save/model.ckpt', global_step=epoch)
        
        writer.close()

train_process(model=model, train=True, input_x=train_data, y_true=train_label, test_data=test_data, test_label=test_label,
              lr=1e-4, dropout=0.5, batch_size=50)
