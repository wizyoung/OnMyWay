# TensorFlow 笔记4 — Graph

<!-- TOC -->

- [TensorFlow 笔记4 — Graph](#tensorflow-笔记4--graph)
    - [1. Graph](#1-graph)
        - [1.1 control_dependencies()](#11-control_dependencies)
    - [2. Graph collection](#2-graph-collection)
        - [2.1 graph.collections](#21-graphcollections)
        - [2.2 add_to_collection()](#22-add_to_collection)
        - [2.3 add_to_collections()](#23-add_to_collections)
        - [2.4 get_collection()](#24-get_collection)
        - [2.5 get_collection_ref()](#25-get_collection_ref)
        - [2.6 get_all_collection_keys()](#26-get_all_collection_keys)
        - [2.7 clear_collection()](#27-clear_collection)
        - [2.8 GraphKeys](#28-graphkeys)

<!-- /TOC -->

## 1. Graph

Graph 部分的源码在 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/ops.py .

一个 Graph 中包含了一系列 Operation 和 Tensor，前者是 TensorFlow 基本计算单元，后者是数据基本单元。

一旦导入 tensorflow，就默认注册了一个 graph，这个 graph 可由 `tf.get_default_graph()`调用获得：

```python
a = tf.constant(1.0)
assert a.graph is tf.get_default_graph()
```

可用 `tf.Graph.as_default()` 的形式生成一个上下文管理器，把指定的 graph 设为默认 graph。

```python
g = tf.Graph()
with g.as_default():
  a = tf.constant(1.0)
  assert a.graph is g
```

**Graph 常用的属性或方法**：

### 1.1 control_dependencies()

pass

## 2. Graph collection

TensorFlow 提供了一个全局存储机制 collection，就是在 Graph 类中以 dict 的形式存储变量的值，这样就不会受到变量名生存空间的影响，且一处保存，随处可取。

首先看 collection 部分源码的结构(ops.py):

```python
class Graph(object):
  
  def __init__(self):
    self._collections = {}
    
  @property
  def collections(self):
    """Returns the names of the collections known to this graph."""
    return list(self._collections)
  
  def add_to_collection(self, name, value):
    _assert_collection_is_ok(name)
    self._check_not_finalized()
    with self._lock:
      if name not in self._collections:
        self._collections[name] = [value]
      else:
        self._collections[name].append(value)
        
  def add_to_collections(self, names, value):
    # 确保 names 里面元素都是不重名的
    names = (names,) if isinstance(names, six.string_types) else set(names)
    for name in names:
      self.add_to_collection(name, value)
    
  def get_collection_ref(self, name):
    _assert_collection_is_ok(name)
    with self._lock:
      coll_list = self._collections.get(name, None)
      if coll_list is None:
        coll_list = []
        self._collections[name] = coll_list
        return coll_list
      
  def get_collection(self, name, scope=None):
    _assert_collection_is_ok(name)
    with self._lock:
      collection = self._collections.get(name, None)
      if collection is None:
        return []
      if scope is None:
        return list(collection)
      else:
        c = []
        regex = re.compile(scope)
        for item in collection:
          if hasattr(item, "name") and regex.match(item.name):
            c.append(item)
        return c
      
  def get_all_collection_keys(self):
    with self._lock:
      return [x for x in self._collections if isinstance(x, six.string_types)]
    
  def clear_collection(self, name):
    self._check_not_finalized()
    with self._lock:
      if name in self._collections:
        del self._collections[name]
```

### 2.1 graph.collections

以 list 形式返回 graph 的 collection 的 name, 也就是 _collections 字典的 key

### 2.2 add_to_collection()

graph.add_to_collection(name, value): 把 value 按指定的 name 添加到 graph 的 collection 中。从上面源码中可以看到，collection 同一 name 可以有多个值，可以把一个 value 反复添加到一个 name 下。

注意，从源码中看到，value 是按 list 形式存起来的。

```python
tf.add_to_collection('a', 1)
print tf.get_collection('a')
# [1]
tf.add_to_collection('a', 2)
print tf.get_collection('a')
# [1, 2]
```

tf.add_to_collection() 与 graph.add_to_collection() 的区别是前者是给默认的 graph 使用的。

### 2.3 add_to_collections()

graph.add_to_collections(names, value): 这里的 names 可以是多个变量名。

```python
# 接上面 2.2 节
# 这里就必须用 get_default_graph() 指定 graph 了
tf.get_default_graph().add_to_collections(['a', 'b'], 'test')
print tf.get_collection('a')
# [2, 1, 'test']
print tf.get_collection('b')
# ['test]
```

### 2.4 get_collection()

graph.get_collection(name, scope=scope): 以 list 形式返回 graph 的 collection 中名为 name 的变量的值。

这里的 scope 用于用正则表达式匹配名为 name 的变量的 name 属性。

同样, tf.get_collection() 是给默认的 graph 使用的。

### 2.5 get_collection_ref()

(graph/tf).get_collection_ref(name): 与 get_collection() 的区别是，若名为 name 的变量不存在，get_collection_ref() 会把这个变量添加到 graph 中，且 value 为[]

### 2.6 get_all_collection_keys()

graph.get_all_collection_keys(): 返回 graph 的 collection 字典的 key names.

### 2.7 clear_collection()

graph.clear_collection(name): 删除 graph 的 collection 中名为 name 的变量。

### 2.8 GraphKeys

TensorFlow 自己也维护一些 collection，比如我们定义的所有 summary op 都会保存在名为 tf.GraphKeys.SUMMARIES 的变量中，比如 tf.Optimizer 的那些子类默认把目标优化变量添加到tf.GraphKeys.TRAINABLE_VARIABLES中，等等。

GraphKeys 是定义在 ops.py 中的一个类，该类定义了一堆 collection 的 key：

```python
class GraphKeys(object):
  GLOBAL_VARIABLES = "variables"
  LOCAL_VARIABLES = "local_variables"
  ...
```

GraphKeys 定义的常见的默认 collection 列表如下：

tf.GraphKeys.GLOBAL_VARIABLES = "variables"：所有变量

tf.GraphKeys.TRAINABLE_VARIABLES = "trainable_variables"：可学习变量

tf.GraphKeys.LOCAL_VARIABLES = "local_variables" 本地变量，常用于临时变量，不用于存储/恢复

tf.GraphKeys.MODEL_VARIABLES = "model_variables" 由 layers 定义的变量

tf.GraphKeys.SUMMARIES = "summaries"

tf.GraphKeys.QUEUE_RUNNERS = "queue_runners"

tf.GraphKeys.MOVING_AVERAGE_VARIABLES = "moving_average_variables"

tf.GraphKeys.REGULARIZATION_LOSSES = "regularization_losses"