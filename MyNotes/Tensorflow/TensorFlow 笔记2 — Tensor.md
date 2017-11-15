# TensorFlow 笔记2 — Tensor

<!-- TOC -->

- [TensorFlow 笔记2 — Tensor](#tensorflow-笔记2--tensor)
    - [1. Tensor 的一些属性](#1-tensor-的一些属性)
    - [2. 一些常见 API](#2-一些常见-api)
        - [2.1 shape 变换](#21-shape-变换)
            - [2.1.1 tf.shape(x) 和 tensor.get_shape()](#211-tfshapex-和-tensorget_shape)
            - [2.1.2 tensor.set_shape() 和 tf.reshape(x)](#212-tensorset_shape-和-tfreshapex)
            - [2.1.3 tf.expand_dims()](#213-tfexpand_dims)
            - [2.1.4 tf.squeeze()](#214-tfsqueeze)
        - [2.2 抽取/切片](#22-抽取切片)
            - [2.2.1 tf.slice()](#221-tfslice)
            - [2.2.2 tf.gather()](#222-tfgather)
            - [2.2.2 tf.split()](#222-tfsplit)
            - [2.2.3 tf.nn.embedding_lookup()](#223-tfnnembedding_lookup)
        - [2.3 拼接](#23-拼接)
            - [2.3.1 tf.concat()](#231-tfconcat)
            - [2.3.2 tf.stack()](#232-tfstack)
        - [2.4 填充](#24-填充)
            - [2.4.1 tf.pad()](#241-tfpad)
            - [2.4.2 tf.tile()](#242-tftile)
        - [2.5 dtype 转换](#25-dtype-转换)
        - [2.6 字符串操作](#26-字符串操作)
        - [2.7 其他](#27-其他)
            - [2.7.1 tf.where()](#271-tfwhere)
            - [2.7.2 tf.convert_to_tensor()](#272-tfconvert_to_tensor)
    - [3. SparseTensor](#3-sparsetensor)

<!-- /TOC -->

## 1. Tensor 的一些属性

Tensor 是 TensorFlow 最基本的数据类型，代表的是 Operation 的输出。

```python
>>> a = tf.constant([1, 2], name='a')
>>> a
<tf.Tensor 'a:0' shape=(2,) dtype=int32>
```

直接终端输入 Tensor 时，能看到 name, shape, dtype 三个属性。

属性列表：

- name: 注意返回的 name 类型是 Unicode 

  ```python
  >>> a.name
  u'a:0'
  ```

  Tensor 命名的形式为"node: src_output"，node 为节点名称，src_output 表示该张量为当前节点的第几个输出。

  对于本例子中的 Tensor，node 为定义时指定的 name。

  对于 c = tf.add(a, b)，则 c 的 name 为"Add:0"。

- shape: 调用 a.shape 返回的值是 TensorShape 对象

  ```python
  >>> a.shape
  TensorShape([Dimension(2)])
  ```

  a.get_shape(): 返回 TensorShape 对象的元组形式

  a.get_shape().as_list(): 将 TensorShape 对象转换成 list 对象

- dtype: Tensorflow 默认整数为 tf.int32, 浮点数为 tf.float32

##2. 一些常见 API

### 2.1 shape 变换

#### 2.1.1 tf.shape(x) 和 tensor.get_shape() 

首先 tensorflow 中有两种 shape: `static (inferred) shape` 和 `dynamic (true) shape`。其中 static shape 一般是由**创建** tensor 时的 op 推断 (infer) 出来的，创建时就确定了；如果 static shape 未定义，则可用 tf.shape() 来获得其 dynamic shape.

tensor.get_shape() 返回的 tuple 中是 static shape, 只有 tensor 有这个方法。tensor.get_shape() 不需要在 session 中即可运行。其实, tensor.get_shape() 就是 tensor.shape 的一个 alias.

tf.shape(x) 需要在 session 中运行，常用于返回 dynamic shape.

区别见示例:

```python
x = tf.placeholder(tf.int32, shape=[4])
print x.get_shape().as_list()
# [4] 这种就是 static shape

y, _ = tf.unique(x)
print y.get_shape().as_list()
# None 这种就是 dynamic shape 了，因为具体的 x 不同，返回的 y 也不同。

z = tf.shape(y)
sess = tf.Session()
print sess.run(z, feed_dict={x: [0, 1, 2, 3]})
# [4] dynamic shape
```

tf.shape 的一个常见应用是:

```python
x = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])
```

在 feed 数据运行后，想知道 None 到底是多少，可通过 tf.shape(x)[0] 的方式获得。

#### 2.1.2 tensor.set_shape() 和 tf.reshape(x)

区别：tensor.set_shape() 更新的是 tensor 的 static shape, 不改变 dynamic shape; 而 tf.reshape(x) 创建的是一个具备不同 dynamic shape 的新的 tensor.

为便于理解，对于 tensor.set_shape(): 

```python
a = tf.constant([1,2,3,4], dtype=tf.int32)
a.set_shape((2,2))
```

上述代码直接报错，提示`ValueError: Shapes (4,) and (2, 2) are not compatible`，因为只有在 tensor 的 static shape 无法确定时，才能用 tensor.set_shape() 更新 shape，而上面的 a 的  static shape 显然是确定的。正确的用是:

```python
b = tf.placeholder(tf.int32, shape=[None, 2, 2, 3])
b.set_shape([1,2,2,3])
```

tf.reshape 的用法:

```
tf.reshape(tensor, shape, name=None)
```

 返回一个具有指定 shape 的新的 tensor。

```python
# tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
# tensor 't' has shape [9]
t1 = tf.reshape(t, [3, 3])
t1.eval()
# [[1, 2, 3],
#  [4, 5, 6],
#  [7, 8, 9]]

# -1 可以用于自动推断 shape
t2 = tf.reshape(t, [-1, 9])
t2.eval()
# [[1, 1, 1, 2, 2, 2, 3, 3, 3],
#  [4, 4, 4, 5, 5, 5, 6, 6, 6]]

# [-1] 可用来展平 tensor
t3 = tf.reshape(t1, [-1])
t3.eval()
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 对一维的 tensor，可用 [] 转换成标量
t4 = tf.reshape(t, [])
t4.eval()
# array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)
```

#### 2.1.3 tf.expand_dims()

tf.expand_dims(input, axis=None, name=None): 类似于 numpy 中的 newaxis，指定位置扩充一个长度为1的轴

```python
# 't' is a tensor of shape [2]
t1 = tf.expand_dims(t, 0)
t1.get_shape().as_list() # [1, 2]
```

#### 2.1.4 tf.squeeze()

tf.squeeze(input, axis=None, name=None): 从 tensor 中删除大小是 1 的维度。

```python
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
t1 = tf.squeeze(t)
t1.shape.as_list()
# [2, 3]
# 默认删除所有的大小为 1 的轴

t2 = tf.squeeze(t, [2, 4])
t2.shape.as_list()
# [1, 2, 3, 1]
# axis 用 list 指定时，删除指定的几个轴
```

### 2.2 抽取/切片

#### 2.2.1 tf.slice()

tf.slice(input_, begin, size, name=None): 按照指定的下标范围抽取**连续**区域的子集。begin 是开始点的坐标，size 是要抽取区域的大小。

```python
# 'input' is [[[1, 1, 1],
#        	   [2, 2, 2]],
#
#       	 [[3, 3, 3],
#             [4, 4, 4]],
#
#            [[5, 5, 5],
#             [6, 6, 6]]]

tf.slice(input, [1, 0, 0], [1, 2, 3]).eval()
# array([[[3, 3, 3],
#         [4, 4, 4]]], dtype=int32)

# -1 表示取全部
tf.slice(input, [1, 0, 0], [1, -1, 3]).eval()
# array([[[3, 3, 3],
#         [4, 4, 4]]])
```

假如要从 input 充抽取 `[[[3, 3, 3], [4, 4, 4]]]`，起点第一个 3 的坐标是`[1, 0, 0]`，要抽取部分的大小在三个维度的长度分别是 1, 2, 3，所以 size = [1, 2, 3]

```python
tf.slice(a, [1, 0, 0], [2, 1, 3]).eval()
# array([[[3, 3, 3]],
# 
#        [[5, 5, 5]]], dtype=int32)
```

#### 2.2.2 tf.gather()

tf.gather(params, indices, validate_indices=None, name=None, axis=0): 按照指定的下标集合从 `axis=0` 中抽取子集，适合抽取`不连续`区域的子集。

看个例子便懂：

```python
# a = [ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
tf.gather(z, [0, 1, 9]).eval()
# array([ 0, 10, 90], dtype=int32)

# 'input' is [[[1, 1, 1],
#        	   [2, 2, 2]],
#
#       	 [[3, 3, 3],
#             [4, 4, 4]],
#
#            [[5, 5, 5],
#             [6, 6, 6]]]
tf.gather(input, [0, 2]).eval()
# array([[[1, 1, 1],
#         [2, 2, 2]],
#
#        [[5, 5, 5],
#         [6, 6, 6]]])
```

#### 2.2.2 tf.split()

tf.split(value, num_or_size_splits, axis=0, num=None, name='split'): 将一个 tensor，沿着 axis 轴 (默认为0)，切割成多份.

```python
# 'value' is a tensor with shape [5, 30]
split0, split1, split2 = tf.split(value, [4, 15, -1], 1)
tf.shape(split0).eval()  # [5, 4]
tf.shape(split1).eval()  # [5, 5]
tf.shape(split2).eval()  # [5, 21]

# 把 value 平均切成三份
split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
# 每一个 split 的 shape 都是[5, 10]
```

#### 2.2.3 tf.nn.embedding_lookup()

tf.nn.embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True, max_norm=None): 由 tf.gather() 导出。功能是按照 ids 顺序返回 params 中的指定部分。

```python
# mat = array([[1, 2, 3],
#        	   [4, 5, 6],
#              [7, 8, 9]])
ids = [1, 2]
res = tf.nn.embedding_lookup(mat, ids)
res.eval()
# array([[4, 5, 6],
#        [7, 8, 9]])
# 其实就是返回第 1 个和第 2 个元素

ids = [[1, 2], [0, 1]]
res = tf.nn.embedding_lookup(mat, ids)
res.eval()
# array([[[4, 5, 6],
#        [7, 8, 9]],
#
#       [[1, 2, 3],
#        [4, 5, 6]]])
# 第一部分是第 1 个和第 2 个元素，第二部分是第 0 个和第 1 个元素
```

### 2.3 拼接

#### 2.3.1 tf.concat()

tf.concat(valuse, axis, name='concat'): 按照指定的**已经存在**的轴进行拼接

```python
# t1 = [[1, 2, 3], [4, 5, 6]]
# t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 0) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 1) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
```

从 shape 的角度看:

```
tf.concat([t1, t2], 0)  # [2,3] + [2,3] ==> [4, 3]
tf.concat([t1, t2], 1)  # [2,3] + [2,3] ==> [2, 6]
```

#### 2.3.2 tf.stack()

tf.stack(values, axis=0, name='stack'): 按照指定的**新建**的轴进行拼接

```python
# t1 = [[1, 2, 3], [4, 5, 6]]
# t2 = [[7, 8, 9], [10, 11, 12]]
tf.stack([t1, t2], 0)  ==> [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
tf.stack([t1, t2], 1)  ==> [[[1, 2, 3], [7, 8, 9]], [[4, 5, 6], [10, 11, 12]]]
tf.stack([t1, t2], 2)  ==> [[[1, 7], [2, 8], [3, 9]], [[4, 10], [5, 11], [6, 12]]]
```

从 shape 的角度看:

```
tf.stack([t1, t2], 0)   # [2,3] + [2,3] ==> [2*,2,3]
tf.stack([t1, t2], 1)   # [2,3] + [2,3] ==> [2,2*,3]
tf.stack([t1, t2], 2)   # [2,3] + [2,3] ==> [2,3,2*]
```

### 2.4 填充

#### 2.4.1 tf.pad()

tf.pad(tensor, paddings, mode='CONSTANT', name=None, constant_values=0) : 在 tensor 周围按照 paddings 的 shape 填充数据。

paddings 是一个 shape 为 [n, 2] 的 tensor，其中 n 表示原 tensor 的秩。对于原输入 tensor 的每一个维度 D，paddings[D, 0] 表示在该维度前面填充的数量，paddings[D, 1] 表示在该维度后面填充的数量。

```python
# 't' is [[1, 2, 3], [4, 5, 6]].
# 'paddings' is [[1, 2], [3, 2]].
tf.pad(t, paddings, 'CONSTANT').eval()
# array([[0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 1, 2, 3, 0, 0],
#        [0, 0, 0, 4, 5, 6, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
```

第一个维度 (行) 前面填充 1 行，后面填充 2 行；第二个维度 (列) 前面填充3 列，后面填充 2 列。

mode 还可选 'REFLECT' 和 'SYMMETRIC'，不太常用。

#### 2.4.2 tf.tile()

tf.tile(input, multiples, name=None): 按照 multiples 把 input 复制平铺。

```python
# a = [[ 1.,  2.],
#      [ 3.,  4.]]
b = tf.tile(a, [1, 3])
# 第 2 个维度上复制3次，第一个维度不操作
# [[ 1.,  2.,  1.,  2.,  1.,  2.],
#  [ 3.,  4.,  3.,  4.,  3.,  4.]]
c = tf.tile(a, [2, 3])
# [[ 1.,  2.,  1.,  2.,  1.,  2.],
#  [ 3.,  4.,  3.,  4.,  3.,  4.],
#  [ 1.,  2.,  1.,  2.,  1.,  2.],
#  [ 3.,  4.,  3.,  4.,  3.,  4.]]
```

### 2.5 dtype 转换

- tf.to_float(x, name='ToFloat'): 返回一个转换为 tf.float32 的 tensor

- tf.to_double(x, name='ToDouble'): 返回一个转换为 tf.float64 的 tensor

- tf.to_int32(x, name='ToInt32'): 返回一个转换为 tf.int32 的 tensor

- tf.to_int64(x, name='ToInt64'): 返回一个转换为 tf.int64 的 tensor

- tf.cast(x, dtype, name=None): 返回一个转换为指定 dtype 的 tensor

- tf.string_to_number(string_tensor, out_type=None, name=None): 将字符串 tensor 转换为 number 类型 (默认 tf.float32) 的 tensor

  ```python
  a = tf.constant('123')
  b = tf.string_to_number(a, tf.int32)
  # b: <tf.Tensor 'StringToNumber_1:0' shape=() dtype=int32>
  ```

TensorFlow 中的 dtype 为 tf.Dtype 类的实例，源码在: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/dtypes.py . 源码通俗易懂。Dtype 类常用的方法有: tf.Dtype.name, tf.Dtype.base_dtype, tf.Dtype.is_floating, tf.Dtype.is_integer, tf.Dtype.as_numpy_dtype 等。

### 2.6 字符串操作

-  tf.string_split(source, delimiter=' '): source 是一维数组，该函数按照 delimiter 将其拆分为多个元素，返回值的类型是 SparseTensor

- tf.string_join(inputs, separator=None, name=None): 拼接字符串

  ```python
  tf.string_join(['hello', 'world'], separator=' ') ==> 'hello world'
  ```

### 2.7 其他

#### 2.7.1 tf.where()

tf.where(condition, x=None, y=None, name=None): 根据条件 condition 返回 x 或者 y 的值。condition 为 bool 型的 Tensor。

```python
x = tf.constant([[1, 2, 3, 4], [3, 1, 2, 1]])
x_ = tf.where(tf.equal(x, 1), x=x, y=tf.zeros_like(x))
# x_ = [[1, 0, 0, 0],
#       [0, 1, 0, 1]]
```

#### 2.7.2 tf.convert_to_tensor()

tf.convert_to_tensor(value, dtype=None, name=None, preferrred_dtype=None)
将 Tensor, ndarray, list, scalar 转换成 tf 的 Tensor 并返回这个新的 Tensor

## 3. SparseTensor

TensorFlow使用三个dense tensor来表达一个sparse tensor：`indices`、`values`、`dense_shape`。

假如我们有一个dense tensor：

```
[[1, 0, 0, 0]
 [0, 0, 2, 0]
 [0, 0, 0, 0]]
```

那么用SparseTensor表达这个数据对应的三个dense tensor如下：

- `indices`：[[0, 0], [1, 2]]
- `values`：[1, 2]
- `dense_shape`：[3, 4]

```python
a = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
```

SparseTensor 转换为 DenseTensor:

- tf.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value=0, validate_indices=True, name=None)
- tf.sparse_tensor_to_dense(sp_input, default_value=0, validate_indices=True, name=None)

上面的 a 转换为 DenseTensor:

```python
b = tf.sparse_to_dense(a.indices, a.dense_shape, a.values)
b.eval()
# array([[1, 0, 0, 0],
#        [0, 0, 2, 0],
#        [0, 0, 0, 0]], dtype=int32)
```

用第一个函数可以用来生成 Onehot 编码：

假如一个 batch 有6个样本，每个样本的 lable 分别是：0, 2, 3, 6, 7, 9:

```python
BATCHSIZE = 6
label = tf.expand_dims(tf.constant([0, 2, 3, 6, 7, 9]), 1)
index = tf.expand_dims(tf.range(0, BATCHSIZE), 1)
_indices = tf.concat([index, label], 1)
# 此时的 indices 为:
# [[0, 0],
#  [1, 2],
#  [2, 3],
#  [3, 6],
#  [4, 7],
#  [5, 9]]
onehot_labels = tf.sparse_to_dense(_indices, [BATCHSIZE, 10], 1.0, 0.0)
onehot_labels.eval()
# array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]], dtype=float32)
```

tf.sparse_tensor_to_dense() 是对 tf.sparse_to_dense() 的一个简单 wrapper, sp_input 为输入的 SparseTensor

```python
c = tf.sparse_tensor_to_dense(a)
# b = tf.sparse_to_dense(a.indices, a.dense_shape, a.values) 的简化版
```


