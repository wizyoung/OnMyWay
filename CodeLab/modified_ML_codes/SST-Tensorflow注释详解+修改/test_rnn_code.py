# coding: utf-8

import tensorflow as tf

# batch_size: 160, 时序长度为32(32个全喂进去才做一次测试), 每次扔进去的特征维度为4096
# 也就是说 100 是 max_time, 代表了 RNN 的记忆时长
# 4096 是输入进去做全连接的 input_size
video_feat = tf.placeholder(tf.float32, [160, 100, 4096], name='video_feat')

# video_feat_mask = tf.placeholder(tf.float32, [2, 32], name='video_feat_mask')

# num_units 实际就是里面全连接神经元个数
# LSTM 输入输出的理解可以参考: https://www.zhihu.com/question/41949741?sort=created
def get_rnn_cell():
    return tf.contrib.rnn.LSTMCell(num_units=128, state_is_tuple=True, initializer=tf.orthogonal_initializer())

# 旧的写法:
# rnn_cell_video = tf.contrib.rnn.LSTMCell(num_units=128, state_is_tuple=True, initializer=tf.orthogonal_initializer())
# print rnn_cell_video.state_size  # LSTMStateTuple(c=128, h=128)
# print rnn_cell_video.output_size  # 128

# multi_rnn_cell_video = tf.contrib.rnn.MultiRNNCell([rnn_cell_video]*2, state_is_tuple=True) 这样写不对!
# 参见: https://stackoverflow.com/questions/48865554/using-dynamic-rnn-with-multirnn-gives-error/49066981#49066981
# https://github.com/tensorflow/tensorflow/issues/14897
# https://github.com/tensorflow/tensorflow/issues/16186
multi_rnn_cell_video = tf.contrib.rnn.MultiRNNCell([get_rnn_cell() for _ in range(2)], state_is_tuple=True)
# print multi_rnn_cell_video.state_size  # (LSTMStateTuple(c=128, h=128), LSTMStateTuple(c=128, h=128))
# print multi_rnn_cell_video.output_size  # 128

# type: tuple, 有2个元素，每个元素是LSTMStateTuple对象, 如:
# LSTMStateTuple(c=<tf.Tensor 'MultiRNNCellZeroState/LSTMCellZeroState/zeros:0' shape=(160, 128) dtype=float32>, 
# h=<tf.Tensor 'MultiRNNCellZeroState/LSTMCellZeroState/zeros_1:0' shape=(160, 128) dtype=float32>)
# c: cell, [batch_size, num_units]
# h: hidden, [batch_size, num_units]
initial_state = multi_rnn_cell_video.zero_state(batch_size=160, dtype=tf.float32)

# sequence_length must be a vector of length batch_size, 其实就是有效长度(考虑到padding)
# 官方说明: Used to copy-through state and zero-out outputs when past a batch element's sequence length. So it's more for correctness than performance
# 更多说明参加: https://blog.csdn.net/u010223750/article/details/71079036
# sequence 里面的元素 应该不超过 max_time

# 输出可参照: https://blog.csdn.net/jerr__y/article/details/61195257
# rnn_outputs: [batch_size, max_time, num_units(128)]
# state: tuple, 包含num_of_rnn 个 final state, 由 c 和 h 组成, 每个的shape是[batch_size, num_units]
rnn_outputs, state = tf.nn.dynamic_rnn(
                    cell=multi_rnn_cell_video, 
                    inputs=video_feat, 
                    initial_state=initial_state,
                    dtype=tf.float32
                )

# sequence_length = tf.reduce_sum(video_feat_mask, axis=-1)
print 
print rnn_outputs
print
print state
