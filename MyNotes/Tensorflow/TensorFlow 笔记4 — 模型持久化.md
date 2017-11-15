# TensorFlow 笔记4 — 模型持久化

<!-- TOC -->

- [TensorFlow 笔记4 — 模型持久化](#tensorflow-笔记4--模型持久化)
    - [1. 简单例子：默认全部载入](#1-简单例子默认全部载入)
        - [1.1 保存模型](#11-保存模型)
        - [1.2 恢复模型](#12-恢复模型)
    - [2. 深入 tf.train.Saver()](#2-深入-tftrainsaver)
        - [2.1  tf.train.Saver():](#21--tftrainsaver)
        - [2.2  saver.save()](#22--saversave)
        - [2.3 saver.restore()](#23-saverrestore)
        - [2.4 saver.export_meta_graph()](#24-saverexport_meta_graph)
    - [3. 灵活应用](#3-灵活应用)
        - [3.1 载入图](#31-载入图)
        - [3.2 保存/加载部分变量](#32-保存加载部分变量)
        - [3.3 图+数据统一存放](#33-图数据统一存放)
        - [3.4 freeze & restore: 根据 ckpt 文件获取变量数据](#34-freeze--restore-根据-ckpt-文件获取变量数据)
        - [3.5 NewCheckpointReader](#35-newcheckpointreader)
        - [3.6 利用预训练的 VGG-16 做迁移学习](#36-利用预训练的-vgg-16-做迁移学习)
    - [4. 深入 MetaGraphDef](#4-深入-metagraphdef)
        - [4.1 MetaInfoDef](#41-metainfodef)
        - [4.2 GraphDef](#42-graphdef)
        - [4.3 SaverDef](#43-saverdef)
        - [4.4 CollectionDef](#44-collectiondef)
    - [5. 一些函数](#5-一些函数)
        - [5.1 tf.train.write_graph()](#51-tftrainwrite_graph)
        - [5.2 tf.train.get_checkpoint_state()](#52-tftrainget_checkpoint_state)

<!-- /TOC -->

## 1. 简单例子：默认全部载入

### 1.1 保存模型

```python
# coding: utf-8
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), dtype=tf.float32, name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), dtype=tf.float32, name='v2')

result = v1 + v2

tf.add_to_collection('var', v1)

init_op = tf.global_variables_initializer()

# 定义一个 Saver() 对象
# 注意，saver 只保存这一句之前的定义的*变量*的值到 data 和 index 文件中，
# 但是变量的声明是存在于 meta 文件中，还是会被保存到 meta 文件中去。
saver = tf.train.Saver()

tf.add_to_collection('var', v1)

with tf.Session() as sess:
    sess.run(init_op)
    print sess.run(result)
    # save函数目的是保存生成 index 和 data 数据文件，但默认一并也把 meta 文件生成了。
    saver.save(sess, 'save1/model.ckpt', global_step=100)
```

保存后 save1目录下将有四个文件：

- checkpoint
- model.ckpt-100.index
- model.ckpt-100.data-00000-of-00001
- model.ckpt-100.meta

其中：

(1) checkpoint 为检查点日志文件，为一个纯文档，里面记录了最新的检查点文件路径，和没被删除的历史检查点文件路径：

```
# save1/checkpoint
model_checkpoint_path: "model.ckpt-101"
all_model_checkpoint_paths: "model.ckpt-101"
```

(2) model.ckpt-100.index 和 model.ckpt-100.data-00000-of-00001 为`数据文件`，是 Google Protobuf 二进制格式的 SSTable 文件，一个为索引，一个为数据，二者结合起来记录了程序中的权重和变量的值，不包含图的结构。

(3) model.ckpt-100.meta 为元图文件，在 TensorFlow 中称为 **MetaGraphDef** 文件 (TensorFlow 中凡是带 def 后缀的都是指的是 Protobuf 格式文件)，包含了图的结构和一些元信息 (版本号，变量如何连接，变量类型声明)。该文件通过调用 tf.train.export_meta_graph() 方法可以生成保存。

### 1.2 恢复模型

```python
# coding: utf-8
import tensorflow as tf

# 使用和保存模型的代码中一样的方式来声明变量。
v1 = tf.Variable(tf.constant(1.0, shape=[1]), dtype=tf.float32, name='v1')
v2 = tf.Variable(tf.constant(3.0, shape=[1]), dtype=tf.float32, name='v2')  # 注意这里改成了3

result = v1 + v2

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)  # 这一步可以不需要
    # 从检查点文件中恢复变量 Variable 的值。restore 目的是从 index 和 data 数据
    # 文件中恢复变量值到当前 Session
    saver.restore(sess, 'save1/model.ckpt-101')  
    print sess.run(result)  # 但是结果是3
```

## 2. 深入 tf.train.Saver()

### 2.1  tf.train.Saver():

tf.train.Saver() 是一个类, 上面的 saver 为它的实例。该部分的源代码在 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/saver.py

Saver 实例化定义时，其初始化参数列表为：

```python
__init__(
    var_list=None,
    reshape=False,
    sharded=False,
    max_to_keep=5,
    keep_checkpoint_every_n_hours=10000.0,
    name=None,
    restore_sequentially=False,
    saver_def=None,
    builder=None,
    defer_build=False,
    allow_empty=False,
    write_version=tf.train.SaverDef.V2,
    pad_step_number=False,
    save_relative_paths=False,
    filename=None
)
```

- var_list: 参数列表，类型为 dict 或 list，使用方法见 3.2。
- reshape: 如果为 True，恢复时可以改变变量的 shape，但总元素个数得一致。
- max_to_keep: checkpoint 历史记录保存数。
- keep_checkpoint_every_n_hours: 每几个小时保存一次 checkpoint 。

### 2.2  saver.save()

```python
save(
    sess,
    save_path,  # 不用加后缀名
    global_step=None,
    latest_filename=None,
    meta_graph_suffix='meta',
    write_meta_graph=True,
    write_state=True
)
```

Saver.save() 函数的作用保存权重和变量这些数据，生成 index 和 data 数据文件。它依赖于一个 Session，保存的是**该 Session 启动的 Graph 中的变量值**。默认情况下，write_meta_graph=True，也就是保存变量值时，一并把 MetaGraphDef 文件也生成，并且后缀默认为 meta，其实就是顺便调用了`export_meta_graph()`函数。

global_step 参数为了方便查看保存的是第几个 step 的 ckpt 文件。类型为 Tensor，或者某个 Tensor 的 name，或者是一个整数。

### 2.3 saver.restore()

```python
restore(
    sess,
    save_path  # 不用加后缀名
)
```

恢复**数据文件**中的变量值到当前 Session 中。save_path 可以手动指定 (不包含后缀名)，如 save1/model.ckpt , 也可调用 latest_checkpoint() 函数来返回。

注意: 在现在的 tf 中，在一个新的代码任务中恢复数据文件中的变量值时，可以不用对之前定义的变量进行初始化了，restore 本身就相当于一种初始化。

### 2.4 saver.export_meta_graph()

```python
export_meta_graph(
    filename=None,
    collection_list=None,
    as_text=False,
    export_scope=None,
    clear_devices=False,
    clear_extraneous_savers=False
)
```

导出保存元图信息，以 MetaGraphDef 文件形式储存。说明：TensorFlow 中凡是后缀带`def`的均指的是 Protobuf 文件。

注意这里 filename 必须是带后缀的完整文件名。配合 as_text=True (否则乱码) 可以将文件保存为 json 等方便查看的文件，主要是方便 debug:

saver.export_meta_graph('save1/model.ckpt.meta.json', as_text=True)

clear_devices: 保存时是否清除 Operation 和 Tensor 的 name 前的设备名。

## 3. 灵活应用

### 3.1 载入图

```python
# coding: utf-8
import tensorflow as tf

# 直接载入图，不用重新定义结构了
saver = tf.train.import_meta_graph('save1/model.ckpt.meta')

with tf.Session() as sess:
    saver.restore(sess, 'save1/model.ckpt')  # 载入所有变量值
    print sess.run(tf.get_default_graph().get_tensor_by_name('add:0'))
    print tf.get_collection('var')
# 输出:
# [ 3.]
# [<tf.Tensor 'v1:0' shape=(1,) dtype=float32_ref>]
```

上面用到了 tf.train.import_meta_graph() 函数：

tf.train.import_meta_graph(meta_graph_or_file, clear_devices=False,  import_scope=None)。该函数接受一个 MetaGraphDef 图协议文件路径作为输入，然后将该 meta 文件的 **graph_def** 中的所有节点(nodes)信息**加载到当前图**。

注意：

(1) 代码中的 get_tensor_by_name 要写完整 Tensor 名字，即 node:src 的格式。

(2) 使用 tf.tran.import_meta_graph() 后，后面使用原模型中的变量时，要初始化，无论是用 saver.restore() 还是用 tf.global_variables_initializer() 都可以，都是加载原模型中的变量值。

### 3.2 保存/加载部分变量

注意到 tf.train.Saver() 在实例化时有一个 var_list 参数，该参数可用来指定保存时保存部分变量，加载时只加载部分变量。注意，加不加这个参数不影响元图信息。

（1）保存时：var_list 只能为 list 形式。

```python
# 保存
saver = tf.train.Saver([v1]) # 只保存 v1 的值到数据文件
# 加载
saver.restore(sess, 'save1/model.ckpt')  # 载入所有变量值
print sess.run(tf.get_default_graph().get_tensor_by_name('v1:0')) # [1.]
print sess.run(tf.get_default_graph().get_tensor_by_name('v2:0')) # 报错，因为没保存
print tf.get_default_graph().get_tensor_by_name('v1:0') # Tensor("v2:0", shape=(1,), dtype=float32_ref) 不影响图结构
```

(2) 加载时: var_list 可为 list 形式或者 dict 形式。

```python
# 保存
saver = tf.train.Saver() # 默认保存所有原图中的变量
# 加载
v1 = tf.Variable(tf.constant(1.0, shape=[1]), dtype=tf.float32, name='new_v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), dtype=tf.float32, name='new_v2')  
saver = tf.train.Saver({'v1': v1, 'v2': tf.get_default_graph().get_tensor_by_name('new_v2:0')})
```

原模型中两个变量的名字为 v1和 v2，新模型中更改为 new_v1 和 new_v2，因此用 dict 形式导入，dict 的 key-value 中，key 为原模型的变量名(op.name), value 为新模型中具体的变量。

如果改成 list 形式: saver = tf.train.Saver([v1, v2]) 则报错，因为：tf.train.Saver([v1, v2]) 本质上等价于 saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})，显然 v1.op.name = 'new_v1'，在原模型中该名字的变量不存在，因此可以改成以下代码形式:

```
saver = tf.train.Saver({v.op.name[4:]: v for v in [v1, v2]})
```

### 3.3 图+数据统一存放

默认情况下，图(meta)和数据(index + data) 是分开存放的，有时候就很不方便。可以借助 graph_util 模块中的 convert_variables_to_constants 函数，将 graph_def 中的变量替换为等同其数值的常量保存，这样保存的文件中既有图结构，又有数据。

(1)保存:

```python
# coding: utf-8
import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0, shape=[1]), dtype=tf.float32, name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), dtype=tf.float32, name='v2')
result = v1 + v2

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 导出当前计算图的 graph_def 部分，只需要这部分就有了图的结构
    graph_def = tf.get_default_graph().as_graph_def()

    # 将图中变量转换为其取值对应的常量。这里只留 add 节点。
    # 函数第三个参数为 node 名的 list，node 名没有后面的 0 。
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])

    # 将转换好的带参数值常量的 graph_def 文件存储成 protobuf 文件，名为 model.pb
    with tf.gfile.GFile('save1/model.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())
```

注意，convert_variables_to_constants()函数中，第三个参数为 output_node_names。这个参数必须指定要保存的 node 的名字。因为 TensorFlow 在存储 graph_def 时，一些系统运算也被存为了节点(如常量声明，变量赋值，加法操作，初始化等)，可通过下面方法查看:

```python
for node in graph_def.node:
  print node.name
# Const
# v1
# v1/Assign
# v1/read
# Const_1
# v2
# v2/Assign
# v2/read
# add
# init
```

其实很多 node 是我们不关心的，就没必要保存的，因此上面只保存了我们关心的名为 add 的 node.

(2) 加载：

```python
import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
  model_filename = 'save1/model.pb'
  # 读取保存的模型文件，并解析为 GraphDef 
  with gfile.FastGFile(model_filename, 'rb') as f:
    graph_def = tf.GraphDef() # 新建一个空的 graph_def
    graph_def.ParseFromString(f.read())
  
  # 加载 graph_def 中保存的图到当前图中。return_elements 中给出了张量的名称，是 node:src 形式的，所以是 add:0
  # result 返回的是 list 形式
  result = tf.import_graph_def(graph_def, return_elements=['add:0'])[0]
  print sess.run(result)
```

import_graph_def(graph_def, return_elements=None): 将 graph_def 中的所有的 Tensor 和 Operation 导入到当前图。当 return_elements 不为空时，把 return_elements 中的 Tensor 和  Operation 返回到前面等号前的变量上，不影响前面的全部  Tensor 和 Operation 导入。

### 3.4 freeze & restore: 根据 ckpt 文件获取变量数据

当拿到别人的 ckpt 文件后(meta + index + data)，可以结合前述方法设计一些简单的 API 来获取自己所需要的原模型中的变量值，方便后期做迁移学习之类使用.

(1) freeze & restore: 储存需要的变量到磁盘，后期再从中获取变量值

第一步：freeze，即存储需要的变量到磁盘中

思路：

- 根据 meta 文件加载旧的 MetaGraphDef，然后获取其 GraphDef 文件
- 启动一个 Session，加载前面 GraphDef 中变量权重参数值
- 只保留我们需要的参量，删除无用部分，生成我们要的 frozen graph_def
- 将 frozen graph_def 保存到磁盘

```python
# coding=utf-8
import tensorflow as tf
import os

def freeze_graph(model_dir, output_node_names):
    '''
    Args:
        model_dir: ckpt 系列文件所在目录
        output_node_names: list, 包含要导出的 node 的名字(不带后面的:0)
    '''

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    output_graph = os.path.join(model_dir, 'frozen_model.pb')
  
    with tf.Session(graph=tf.Graph()) as sess:
        # 导入 MetaGraphDef 到当前 Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

        # 恢复变量值到当前 Session
        saver.restore(sess, input_checkpoint)

        # 转换为包含常量形式的变量值的 GraphDef
        output_graph_def = \
        tf.graph_util.convert_variables_to_constants(
            sess, 
            tf.get_default_graph().as_graph_def(), 
            output_node_names=output_node_names)

        # 储存
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())    
```

保存后，我们将得到一个`frozen_model.pb`的文件。

(2) restore: 

```python
# coding:utf-8

import tensorflow as tf

def load_graph(frozen_graph_filename):

    # 从磁盘加载 frozen_model.pb 文件
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 返回的时候默认给每个 node 名字前加 'prefix'，
    # 避免对当前 graph 中的变量产生影响.
    # 注意：如果不设定 name, tf 自动加前缀 'import'
    # 返回一个 graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='prefix')
    return graph

graph = load_graph('/Users/chenyang/Documents/pycharm project/learn_tf_201711/save1/frozen_model.pb')
v1 = graph.get_tensor_by_name('prefix/add:0')

with tf.Session(graph=graph) as sess:
    print sess.run(v1)
```

这里先 freeze 再 restore，即先储存再加载。当然也可以修改前面两个函数，直接读取 meta 文件，加载变量，转换，最后输出到当前任务代码中。

这一部分，TensorFlow 也提供了一个叫 [freeze graph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) 的工具，实现的是同样的功能。

### 3.5 NewCheckpointReader

TensorFlow 中还提供了一种非常简单的模式来根据检查点文件获取其中的变量值：tf.train.NewCheckpointReader()。

```python
...
# 函数里面填写检查点文件路径名
reader = tf.train.NewCheckpointReader('save1/model.ckpt')
all_vars = reader.get_variable_to_shape_map()

print all_vars
print reader.get_tensor('v1')  # 1.0
```

返回的 all_vars 是一个 dict，里面每个元素的 key 为变量名(不带后面的 src 号)，value 为 shape。上面代码的 all_vars 返回结果为: `{'v1': [1], 'v2': [1]}`。可用 reader.get_tensor() 直接获取变量数值。

### 3.6 利用预训练的 VGG-16 做迁移学习

看代码:

```python
import tensorflow as tf

# 载入 MetaGraph 到当前图
vgg_saver = tf.train.import_meta_graph('xxx/vgg/results/vgg-16.meta')

vgg_graph = tf.get_default_graph()

# 输入
self.x_plh = vgg_graph.get_tensor_by_name('input:0')

# 选择中间一个节点
output_conv =vgg_graph.get_tensor_by_name('conv1_2:0')

# 冻结前面部分，不让梯度反向传播回去，阻止训练
output_conv_sg = tf.stop_gradient(output_conv)

# 继续后面的操作
output_conv_shape = output_conv_sg.get_shape().as_list()
W1 = tf.get_variable('W1', shape=[1, 1, output_conv_shape[3], 32], \
    initializer=tf.random_normal_initializer(stddev=1e-1))
b1 = tf.get_variable('b1', shape=[32], initializer=tf.constant_initializer(0.1))
z1 = tf.nn.conv2d(output_conv_sg, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
a = tf.nn.relu(z1)
```

## 4. 深入 MetaGraphDef

2.4 中写到，export_meta_graph() 函数可以将图的 MetaGraphDef 文件导出，我们来看看 MetaGraphDef 文件的结构：

MetaGraphDef 中包含以下几部分:

- MetaInfoDef：记录了 TF 计算图中的元数据和所有用到的方法(op)的信息
- GraphDef: 记录节点(node)信息，也就是 node 之间如何连接的。
- SaverDef: 记录了模型持久化时需要的一些参数
- CollectionDef: 记录 collection 信息

### 4.1 MetaInfoDef

MetaInfoDef Protobuf 的定义:

```json
message MetaInfoDef{
  string meta_graph_version = 1;  // 计算图版本号
  Oplist stripped_op_list = 2;  // 图中用到的所有运算方法的信息
  google.protobuf.Any any_info =3;
  repeated string tags = 4;  // 用户指定的标签
}
```

 MetaInfoDef 中 stripped_op_list 记录了图中用到的所有运算(op)方法的信息，如果某一个运算出现了多次，在 stripped_op_list 中只记录一次。stripped_op_list 里面的元素是 OpDef，每一个 OpDef 记录了 op 的详细信息。如 Add 这个 op：

``` json
op {
      name: "Add"  // 名称
      input_arg {  // 输入
        name: "x"
        type_attr: "T"
      }
      input_arg {
        name: "y"
        type_attr: "T"
      }
      output_arg {  // 输出
        name: "z"
        type_attr: "T"
      }
      attr {  // 一些其他属性
        name: "T"
        type: "type"
        allowed_values {
          list {
            type: DT_HALF
            type: DT_FLOAT
            type: DT_DOUBLE
            ...
          }
        }
      }
   }
```

### 4.2 GraphDef

GraphDef 主要记录的是运算图中的节点(node)信息。因为 MetaInfoDef 中已经包含了所有运算的具体信息，因此 GraphDef 只关注元素的**连接结构**。

GraphDef 里面的元素主要是 NodeDef，GraphDef 的定义为:

```json
message GraphDef{
  repeated NodeDef node = 1;  
  VersionDef versions = 2;  
}
message NodeDef{
  string name = 1;  // 节点(node)名称
  string op = 2;  //  node 使用的运算方法名称
  repeated string input = 3; // node 使用的运算的输入的名称，为 node:src_output 格式，当 src_output 为0时一般省略掉:0
  string device = 4;  // 设备(CPU/GPU)，为空时 TF 自动选取
  map<string, AttrValue> attr = 5;  // 其他配置信息
}
```

比如 v1 这个 Variable node:

```json
node {
    name: "v1"
    op: "VariableV2"
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
            dim {
              size: 1
            }
          }
        }
      }
    }
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_FLOAT
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 1
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
  }
```

### 4.3 SaverDef

SaverDef 中主要记录了模型持久化时需要的一些参数，如保存操作和加载操作在 GraphDef 中的名称，保存的频率，清理历史记录等。因为  SaverDef 比较简单，我们直接看这部分文件的信息：

```json
saver_def {
  filename_tensor_name: "save/Const:0"
  save_tensor_name: "save/control_dependency:0"
  restore_op_name: "save/restore_all"
  max_to_keep: 5
  keep_checkpoint_every_n_hours: 10000.0
  version: V2
}
```

save 保存这个操作在 GraphDef 中对应的 node 的 name 就是 save/control_dependency，restore 对应 save/restore_all。

### 4.4 CollectionDef

TensorFlow 中的图中可以维护不同集合(collections)，其底层实现就是通过 CollectionDef 这个属性：

```json
collection_def {
  key: "trainable_variables"
  value {
    bytes_list {
      value: "\n\004v1:0\022\tv1/Assign\032\tv1/read:0"
      value: "\n\004v2:0\022\tv2/Assign\032\tv2/read:0"
      value: "\n\004v3:0\022\tv3/Assign\032\tv3/read:0"
    }
  }
}
collection_def {
  key: "variables" // 所有变量的集合
  value {
    bytes_list {
      value: "\n\004v1:0\022\tv1/Assign\032\tv1/read:0"
      value: "\n\004v2:0\022\tv2/Assign\032\tv2/read:0"
      value: "\n\004v3:0\022\tv3/Assign\032\tv3/read:0"
    }
  }
}
```

## 5. 一些函数

### 5.1 tf.train.write_graph()

```python
write_graph(
    graph_or_graph_def,
    logdir,
    name,
    as_text=True
)
```

第一个参数可以是 graph 或者 graph_def, 第二个参数时保存的路径(不包括最后带后缀的文件名)，name 是文件名。这个函数是把 graph_def 以文本形式保存（也就是只有图结构信息，不含权重）

### 5.2 tf.train.get_checkpoint_state()

```python
get_checkpoint_state(
    checkpoint_dir,
    latest_filename=None
)
```

checkpoint_dir 为保存的检查点文件的文件夹路径。该函数自动搜索 checkpoint_dir 中的 checkpoint 文本文件，设返回值为 checkpoint。则: checkpoint.model_checkpoint_path 和 checkpoint.all_model_checkpoint_paths 则对应最新的检测点文件和所有检查点文件名称。






