import tensorflow as tf
from cytf.layer import conv2d
from pprint import pprint

input = tf.get_variable('x', shape=[100, 32, 32, 3])
net = conv2d(input, 64, [3, 3], stride=[1, 1], padding='same', name='layer1_left/conv2', regualizer={'l2':0.004}, 
             BN=0.999, ema=True, initializer='he_normal')
net = conv2d(net, 128, [3, 3], stride=[1, 1], padding='same', name='layer2_left/conv2', regualizer={'l2':0.004}, 
             BN=True, ema=False, initializer='he_uniform', activation='relu')

net_ = conv2d(input, 64, [3, 3], stride=[1, 1], padding='same', name='layer1_right/conv2', regualizer={'l2':0.004}, 
              BN=True, ema=False, activation=None)
net_ = conv2d(net_, 128, [3, 3], stride=[1, 1], padding='same', name='layer2_right/conv2', regualizer={'l2':0.004}, 
              BN=True, ema=False, activation='tanh')

print net_.name

with tf.name_scope('output'):
    output = tf.concat([net, net_], axis=0, name='combine')

print output.name

writer = tf.summary.FileWriter('./log', tf.get_default_graph())
writer.close()

print tf.get_default_graph().get_all_collection_keys()
print '===' * 10
pprint(tf.get_collection('losses'))
print '---' * 10
pprint(tf.get_collection('variables'))
print '---' * 10
pprint(tf.get_collection('trainable_variables'))
print '---' * 10
pprint(tf.get_collection('ema'))