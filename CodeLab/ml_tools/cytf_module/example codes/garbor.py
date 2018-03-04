# coding: utf-8
mport tensorflow as tf
import os
import numpy as np
from cytf.layer import conv2d, dense, flatten, max_pool, activation
from cytf.initializers import he_normal

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = p.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y

def load_CIFAR(Foldername):
    train_data = np.zeros([50000,32,32,3])
    train_label = np.zeros([50000,10])
    
    for sample in range(5):
        X,Y = load_CIFAR_batch(Foldername+"/data_batch_"+str(sample+1))
        for i in range(3):
            train_data[10000*sample:10000*(sample+1),:,:,i] = X[:,i,:,:]
        for i in range(10000):
            train_label[i+10000*sample][Y[i]] = 1
    
    test_data = np.zeros([10000,32,32,3])
    test_label = np.zeros([10000,10])
    X,Y = load_CIFAR_batch(Foldername+"/test_batch")
    for i in range(3):
        test_data[0:10000,:,:,i] = X[:,i,:,:]
    for i in range(10000):
        test_label[i][Y[i]] = 1

    train_mean = np.mean(train_data, axis=0)
    test_mean = np.mean(test_data, axis=0)

    train_data = train_data - train_mean
    test_data = test_data - test_mean
    
    return train_data, train_label, test_data, test_label

def Expand_dim_down(input, num):
    res = input
    for i in range(1, num + 1):
        res = tf.expand_dims(res, i)
    return res

def Expand_dim_up(input, num):
    res = input
    for i in range(num):
        res = tf.expand_dims(res, 0)
    return res

def Gabor_filter(Theta, Lambda, Fai, Sigma, Gamma, size, in_channel, out_channel):
    coordinate_begin = - (size - 1) / 2.0
    coordinate_end = - coordinate_begin
    tmp = tf.linspace(coordinate_begin, coordinate_end, size)
    tmp = Expand_dim_down(tmp, 3)
    x = tf.tile(tmp, [size, 1, in_channel, out_channel])
    x = tf.reshape(x, [size, size, in_channel, out_channel])
    y = tf.tile(tmp, [1, size, in_channel, out_channel])

    Theta = tf.reshape(tf.tile(tf.expand_dims(Theta, 0), [6, 1]), [-1])
    Lambda = tf.reshape(tf.tile(tf.expand_dims(Lambda, 1), [1, 6]), [-1])

    Theta = tf.tile(Expand_dim_up(Theta, 3), [size, size, in_channel, 1])
    Lambda = tf.tile(Expand_dim_up(Lambda, 3), [size, size, in_channel, 1])

    Sigma = tf.multiply(0.56, Lambda)

    x_ = tf.add(tf.multiply(x, tf.cos(Theta)), tf.multiply(y, tf.sin(Theta)))
    y_ = tf.add(-tf.multiply(x, tf.sin(Theta)), tf.multiply(y, tf.cos(Theta)))
    pi = tf.asin(1.0) * 2

    res = tf.multiply(tf.exp(-tf.div(tf.add(tf.square(x_), tf.square(y_)), tf.multiply(2.0, tf.square(Sigma)))), tf.cos(2.0 * pi * x_ / Lambda))
    print tf.shape(res).eval()
    return res

print 'reading data...'
train_data, train_label, test_data, test_label = load_CIFAR("dataset")
print 'reading data finish'


batch_size = 128
x = tf.placeholder("float", shape=[batch_size, 32, 32, 3])
y = tf.placeholder("float", shape=[batch_size, 10])
# keep_prob = tf.placeholder("float")

#Gabor
with tf.variable_scope('layer1/garbor'):
    pi = tf.asin(1.0) * 2.0
    Theta = tf.Variable(tf.linspace(0.0, pi * 1.5, 6))
    #print Theta.eval()
    Lambda = tf.Variable(tf.linspace(2.0, 6.0, 6))
    #print Lambda.eval()
    Fai = tf.constant([0.0])
    Sigma = 0.56 * Lambda
    Gamma = tf.constant([1.0])
    Gabor = Gabor_filter(Theta, Lambda, Fai, Sigma, Gamma, 11, 3, 36)
    Gabor_conv = tf.nn.conv2d(x, Gabor, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(tf.layers.batch_normalization(inputs=convolution_2d(x, Gabor, b_conv1), axis=0, training=True))
    pool1 = max_pooling(conv1, 2, 2)

conv2 = conv2d(pool1, [3, 3], 1, 'same', initializer='he_normal', name='layer2/conv2',
               BN=True, regualizer={'l2': 4e-4})
pool2 = max_pool(conv2, [2, 2], 2, 'valid', name='layer2/maxpool')

conv3 = conv2d(pool2, [3, 3], 1, 'same', initializer='he_normal', name='layer3/conv2',
               BN=True, regualizer={'l2': 4e-4})
pool3 = max_pool(conv3, [2, 2], 2, 'valid', name='layer2/maxpool')

flattened = flatten(pool3, name='flatten')

net = dense(flattened, 128, name='layer3/dense', initializer='he_normal', BN=True, 
            regualizer={'l2': 4e-4}, activation='relu')
y_pred = dense(net, 10, name='layer4/dense', initializer='he_normal', BN=True, 
            regualizer={'l2': 4e-4})
            
with tf.name_scope('loss_function'):
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
        tf.add_to_collection('losses', cross_entropy)
        losses = tf.add_n(tf.get_collection('losses'), name='total_loss')

with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(losses)

with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

correct_num = tf.reduce_sum(tf.cast(correct_prediction, "float"))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    batch_image = np.zeros([batch_size,32,32,3])
    batch_label = np.zeros([batch_size,10])
    
    f = open('Gabor_CNN_adam1e-3.txt', 'a')

    for i in range(50):
        num = 0
        total = 0
        for j in range(50000 / batch_size):
            batch_image = train_data[j*batch_size : (j+1)*batch_size, :, :, :]
            batch_label = train_label[j*batch_size : (j+1)*batch_size, :]

            sess.run([train_step], feed_dict={x: batch_image, y: batch_label})
            num = num + sess.run(correct_num, (feed_dict={x: batch_image, y: batch_label}))
            total = total + batch_size

            if j%50 == 49:
                print "step %d, %d / 50000, training accuracy %f"%(i, total, num/total)
                #print W_conv1.eval()
                print sess.run(Theta), sess.run(Lambda)
                plt.figure("show")
                img = np.zeros([32, 32, 3])
                show = np.zeros([32, 32, 3])
                img = batch_image[3, :, :, :]
                plt.imshow(img)
                #plt.show()

                tmp = sess.run(Gabor_conv, feed_dict={x: batch_image, y: batch_label})
                img0 = np.zeros([32, 32])
                img0 = tmp[3, :, :, 4]
                plt.imshow(img0, cmap = cm.gray)
                #plt.show()


        f.write("step %d, training accuracy %f"%(i, num/total))
        f.write('\n')

        num = 0
        total = 0

        for j in range(10000 / batch_size):
            batch_image = test_data[j*batch_size : (j+1)*batch_size, :, :, :]
            batch_label = test_label[j*batch_size : (j+1)*batch_size, :]

            num = num + sess.run(correct_num, feed_dict={x: batch_image, y: batch_label})
            total = total + batch_size

        print "step %d, %d / 10000, test accuracy %f"%(i, total, num/total)
        f.write("step %d, test accuracy %f" % (i, num / total))
        f.write('\n')