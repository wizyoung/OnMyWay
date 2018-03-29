# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

from yolo.net.net import Net 

class YoloNet(Net):

  def __init__(self, common_params, net_params, test=False):
    """
    common params: a params dict
    net_params   : a params dict
    """
    super(YoloNet, self).__init__(common_params, net_params)
    #process params
    self.image_size = int(common_params['image_size'])
    self.num_classes = int(common_params['num_classes'])
    self.cell_size = int(net_params['cell_size'])
    self.boxes_per_cell = int(net_params['boxes_per_cell'])
    self.batch_size = int(common_params['batch_size'])
    self.weight_decay = float(net_params['weight_decay'])

    if not test:
      self.object_scale = float(net_params['object_scale'])
      self.noobject_scale = float(net_params['noobject_scale'])
      self.class_scale = float(net_params['class_scale'])
      self.coord_scale = float(net_params['coord_scale'])

  def inference(self, images):
    """Build the yolo model

    Args:
      images:  4-D tensor [batch_size, image_height, image_width, channels]
    Returns:
      predicts: 4-D tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
    """
    conv_num = 1
    temp_conv = self.conv2d('conv' + str(conv_num), images, [7, 7, 3, 64], stride=2)
    conv_num += 1


    temp_pool = self.max_pool(temp_conv, [2, 2], 2)

    temp_conv = self.conv2d('conv' + str(conv_num), temp_pool, [3, 3, 64, 192], stride=1)
    conv_num += 1

    temp_pool = self.max_pool(temp_conv, [2, 2], 2)

    temp_conv = self.conv2d('conv' + str(conv_num), temp_pool, [1, 1, 192, 128], stride=1)
    conv_num += 1
    
    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 128, 256], stride=1)
    conv_num += 1

    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [1, 1, 256, 256], stride=1)
    conv_num += 1

    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 512], stride=1)
    conv_num += 1

    temp_conv = self.max_pool(temp_conv, [2, 2], 2)

    for i in range(4):
      temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [1, 1, 512, 256], stride=1)
      conv_num += 1

      temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 512], stride=1)
      conv_num += 1

    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [1, 1, 512, 512], stride=1)
    conv_num += 1

    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 1024], stride=1)
    conv_num += 1

    temp_conv = self.max_pool(temp_conv, [2, 2], 2)

    for i in range(2):
      temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [1, 1, 1024, 512], stride=1)
      conv_num += 1

      temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 1024], stride=1)
      conv_num += 1

    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
    conv_num += 1

    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=2)
    conv_num += 1

    #
    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
    conv_num += 1
    
    temp_conv = self.conv2d('conv' + str(conv_num), temp_conv, [3, 3, 1024, 1024], stride=1)
    conv_num += 1


    #Fully connected layer
    local1 = self.local('local1', temp_conv, 49 * 1024, 4096)


    local1 = tf.nn.dropout(local1, keep_prob=0.5)

    local2 = self.local('local2', local1, 4096, self.cell_size * self.cell_size * ( self.num_classes + 5 * self.boxes_per_cell), leaky=False)

    # N * 7 * 7 * 20(10 + 5 * 2)
    # 10为分类，10-12为box置信度，最后面是坐标
    local2 = tf.reshape(local2, [tf.shape(local2)[0], self.cell_size, self.cell_size, self.num_classes + 5 * self.boxes_per_cell])

    predicts = local2


    return predicts

  def iou(self, boxes1, boxes2):
    """calculate ious: 全图box1与输入的一个标签box2的iou
    Args:
      boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)
    Return:
      iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    """
    # tf.pack ==> tf.stack for API changes, so load weights might occur errors
    # [xmin, ymin, xmax, ymax]
    # stack后坐标变为[4, 7, 7, 2]
    boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                      boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
    # shape变成[7, 7, 2, 4]
    boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
    # shape: label形式的[xmin, ymin, xmax, ymax]
    boxes2 =  tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                      boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])

    #calculate the left up point
    # shape [7, 7, 2, 2]
    lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
    rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

    #intersection, [7, 7, 2, 2]
    intersection = rd - lu 

    # shape: [7,7,2]
    inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

    # 有效限定条件
    mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)
    
    # 有效 intersection 矩阵 []
    inter_square = mask * inter_square
    
    #calculate the boxs1 square and boxs2 square
    square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
    square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])
    
    # 返回值shape: [7, 7, 2]
    return inter_square/(square1 + square2 - inter_square + 1e-6)

  def cond1(self, num, object_num, loss, predict, label, nilboy):
    """
    if num < object_num
    """
    return num < object_num


  def body1(self, num, object_num, loss, predict, labels, nilboy):
    """
    calculate loss
    Args:
      predict: 3-D tensor [cell_size, cell_size, 5 * boxes_per_cell]
      labels : [max_objects, 5]  (x_center, y_center, w, h, class)
    """
    label = labels[num:num+1, :]
    label = tf.reshape(label, [-1])

    # label: [xcenter, ycenter, box_w, box_h, class_num]

    #calculate objects  tensor [CELL_SIZE, CELL_SIZE]
    # min_x是在最后的7*7图上的位置
    min_x = (label[0] - label[2] / 2) / (self.image_size / self.cell_size)
    max_x = (label[0] + label[2] / 2) / (self.image_size / self.cell_size)

    min_y = (label[1] - label[3] / 2) / (self.image_size / self.cell_size)
    max_y = (label[1] + label[3] / 2) / (self.image_size / self.cell_size)

    min_x = tf.floor(min_x)
    min_y = tf.floor(min_y)

    max_x = tf.ceil(max_x)
    max_y = tf.ceil(max_y)

    temp = tf.cast(tf.stack([max_y - min_y, max_x - min_x]), dtype=tf.int32)
    objects = tf.ones(temp, tf.float32)

    temp = tf.cast(tf.stack([min_y, self.cell_size - max_y, min_x, self.cell_size - max_x]), tf.int32)
    temp = tf.reshape(temp, (2, 2))
    objects = tf.pad(objects, temp, "CONSTANT")

    # 通过以上计算，objects 为 7 * 7 掩膜矩阵，含有目标的网格标为1，其余为0

    #calculate objects  tensor [CELL_SIZE, CELL_SIZE]
    #calculate responsible tensor [CELL_SIZE, CELL_SIZE]
    center_x = label[0] / (self.image_size / self.cell_size)
    center_x = tf.floor(center_x)

    center_y = label[1] / (self.image_size / self.cell_size)
    center_y = tf.floor(center_y)

    response = tf.ones([1, 1], tf.float32)

    temp = tf.cast(tf.stack([center_y, self.cell_size - center_y - 1, center_x, self.cell_size - center_x - 1]), tf.int32)
    temp = tf.reshape(temp, (2, 2))
    response = tf.pad(response, temp, "CONSTANT")

    # 通过以上计算, response 为 7 * 7 掩膜矩阵，含有中心点的那一个网格标为1，其余为0

    #calculate iou_predict_truth [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    predict_boxes = predict[:, :, self.num_classes + self.boxes_per_cell:]
    
    # 7 * 7 * 2 * 4 坐标
    predict_boxes = tf.reshape(predict_boxes, [self.cell_size, self.cell_size, self.boxes_per_cell, 4])

    # 输出都是0-1之间。
    # 前两个是中心点坐标，要乘以网格大小
    # 后两个是长和宽，要乘以全图边长
    predict_boxes = predict_boxes * [self.image_size / self.cell_size, self.image_size / self.cell_size, self.image_size, self.image_size]

    base_boxes = np.zeros([self.cell_size, self.cell_size, 4])

    for y in range(self.cell_size):
      for x in range(self.cell_size):
        #基准网格的坐标
        base_boxes[y, x, :] = [self.image_size / self.cell_size * x, self.image_size / self.cell_size * y, 0, 0]
    # shape: [7, 7, 2, 4] 存储了计算中心点绝对坐标需要的网格相对坐标
    base_boxes = np.tile(np.resize(base_boxes, [self.cell_size, self.cell_size, 1, 4]), [1, 1, self.boxes_per_cell, 1])

    # 中心点坐标转变成整图绝对坐标了
    predict_boxes = base_boxes + predict_boxes

    # 返回IOU值，shape: [7, 7, 2]
    iou_predict_truth = self.iou(predict_boxes, label[0:4])
    #calculate C [cell_size, cell_size, boxes_per_cell]
    # 只对response有效的区域感兴趣
    # C 是置信度标签，为 IOU * response, shape: [7, 7, 2]
    C = iou_predict_truth * tf.reshape(response, [self.cell_size, self.cell_size, 1])

    #calculate I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    I = iou_predict_truth * tf.reshape(response, (self.cell_size, self.cell_size, 1))
    
    # keep_dims is deprecated, use keepdims instead
    # 计算 2 个 box 中哪个 置信度 大, 保留大的值。shape: [7, 7, 1]
    max_I = tf.reduce_max(I, 2, keepdims=True)

    # TODO: * 后面多余了，但是也不影响结果
    # shape: [7, 7, 1] 为 1 的位置为最大 置信度 的位置, 保存的是对应位置2个box中的最大置信度
    I = tf.cast((I >= max_I), tf.float32) * tf.reshape(response, (self.cell_size, self.cell_size, 1))

    #calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    # 为 1 的位置为不是 max_置信度 的位置
    no_I = tf.ones_like(I, dtype=tf.float32) - I 

    # 预测的每个格子的置信度: [7, 7, 2]
    p_C = predict[:, :, self.num_classes:self.num_classes + self.boxes_per_cell]

    #calculate truth x,y,sqrt_w,sqrt_h 0-D
    x = label[0]
    y = label[1]

    sqrt_w = tf.sqrt(tf.abs(label[2]))
    sqrt_h = tf.sqrt(tf.abs(label[3]))
    #sqrt_w = tf.abs(label[2])
    #sqrt_h = tf.abs(label[3])

    #calculate predict p_x, p_y, p_sqrt_w, p_sqrt_h 3-D [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    # [7, 7, 2]
    p_x = predict_boxes[:, :, :, 0]
    p_y = predict_boxes[:, :, :, 1]

    # 预测的框的长宽的开方，加了限定条件: [7, 7, 2]
    p_sqrt_w = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
    p_sqrt_h = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))
    #calculate truth p 1-D tensor [NUM_CLASSES]
    P = tf.one_hot(tf.cast(label[4], tf.int32), self.num_classes, dtype=tf.float32)

    #calculate predict p_P 3-D tensor [CELL_SIZE, CELL_SIZE, NUM_CLASSES]
    # 分类信息
    p_P = predict[:, :, 0:self.num_classes]

    #class_loss, 分类损失
    class_loss = tf.nn.l2_loss(tf.reshape(objects, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale
    #class_loss = tf.nn.l2_loss(tf.reshape(response, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale

    #object_loss
    object_loss = tf.nn.l2_loss(I * (p_C - C)) * self.object_scale
    #object_loss = tf.nn.l2_loss(I * (p_C - (C + 1.0)/2.0)) * self.object_scale

    #noobject_loss
    #noobject_loss = tf.nn.l2_loss(no_I * (p_C - C)) * self.noobject_scale
    # TODO: 把这项删掉试试? 可以删吗？好像不可以
    noobject_loss = tf.nn.l2_loss(no_I * (p_C)) * self.noobject_scale

    #coord_loss
    coord_loss = (tf.nn.l2_loss(I * (p_x - x)/(self.image_size/self.cell_size)) +
                 tf.nn.l2_loss(I * (p_y - y)/(self.image_size/self.cell_size)) +
                 tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w))/ self.image_size +
                 tf.nn.l2_loss(I * (p_sqrt_h - sqrt_h))/self.image_size) * self.coord_scale

    nilboy = I

    return num + 1, object_num, [loss[0] + class_loss, loss[1] + object_loss, loss[2] + noobject_loss, loss[3] + coord_loss], predict, labels, nilboy


  def loss(self, predicts, labels, objects_num):
    """Add Loss to all the trainable variables

    Args:
      predicts: 4-D tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
      ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
      labels  : 3-D tensor of [batch_size, max_objects, 5]
      objects_num: 1-D tensor [batch_size]
    """
    class_loss = tf.constant(0, tf.float32)
    object_loss = tf.constant(0, tf.float32)
    noobject_loss = tf.constant(0, tf.float32)
    coord_loss = tf.constant(0, tf.float32)
    loss = [0, 0, 0, 0]
    for i in range(self.batch_size):
      predict = predicts[i, :, :, :]
      label = labels[i, :, :]
      object_num = objects_num[i] # 第 i 张图上的目标框数量
      nilboy = tf.ones([7,7,2])
      # cond1 and body1 params:
      # num, object_num, loss, predict, label, nilboy
      # repeat body1 while cond1 is true
      # 预测的框的输出依次去与图中每个 ground truth bbox 比较，计算损失
      tuple_results = tf.while_loop(self.cond1, self.body1, [tf.constant(0), object_num, [class_loss, object_loss, noobject_loss, coord_loss], predict, label, nilboy])
      for j in range(4):
        loss[j] = loss[j] + tuple_results[2][j]
      nilboy = tuple_results[5]

    tf.add_to_collection('losses', (loss[0] + loss[1] + loss[2] + loss[3])/self.batch_size)

    tf.scalar_summary('class_loss', loss[0]/self.batch_size)
    tf.scalar_summary('object_loss', loss[1]/self.batch_size)
    tf.scalar_summary('noobject_loss', loss[2]/self.batch_size)
    tf.scalar_summary('coord_loss', loss[3]/self.batch_size)
    # weight_loss是什么？
    tf.scalar_summary('weight_loss', tf.add_n(tf.get_collection('losses')) - (loss[0] + loss[1] + loss[2] + loss[3])/self.batch_size )

    # 返回总损失和nilboy
    # nilboy: [7, 7, 1], 保存的是对应位置2个box中的最大置信度
    return tf.add_n(tf.get_collection('losses'), name='total_loss'), nilboy