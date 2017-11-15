# TensorFlow 笔记3 — Session

<!-- TOC -->

- [TensorFlow 笔记3 — Session](#tensorflow-笔记3--session)
    - [1. Session](#1-session)
    - [2. InteractiveSession](#2-interactivesession)

<!-- /TOC -->

Session 部分的源码在：https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/client/session.py ，源码大部分还是通俗易懂的。

可以看到，Session 类的继承关系为: Session —> BaseSession —> SessionInterface

## 1. Session

class Session 部分的源码:

```python
class Session(BaseSession):

  def __init__(self, target='', graph=None, config=None):
    super(Session, self).__init__(target, graph, config=config)
    self._default_graph_context_manager = None
    self._default_session_context_manager = None

  def __enter__(self):
    if self._default_graph_context_manager is None:
      self._default_graph_context_manager = self.graph.as_default()
    else:
      raise RuntimeError('Session context managers are not re-entrant. '
                         'Use `Session.as_default()` if you want to enter '
                         'a session multiple times.')
    if self._default_session_context_manager is None:
      self._default_session_context_manager = self.as_default()
    self._default_graph_context_manager.__enter__()
    return self._default_session_context_manager.__enter__()

  def __exit__(self, exec_type, exec_value, exec_tb):
    if exec_type is errors.OpError:
      logging.error('Session closing due to OpError: %s', (exec_value,))
    self._default_session_context_manager.__exit__(
        exec_type, exec_value, exec_tb)
    self._default_graph_context_manager.__exit__(exec_type, exec_value, exec_tb)

    self._default_session_context_manager = None
    self._default_graph_context_manager = None

    self.close()

  @staticmethod
  def reset(target, containers=None, config=None):
    if target is not None:
      target = compat.as_bytes(target)
    if containers is not None:
      containers = [compat.as_bytes(c) for c in containers]
    else:
      containers = []
    tf_session.TF_Reset(target, containers, config)
```

Session 的一些属性和方法:

**属性**：

- graph: 返回这个 session 所在的 graph。

  其实 graph 是一个 property 函数：

  ```python
  # 这一部分定义在 class BaseSession 中

  if graph is None:
    # ops 模块在 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/ops.py, 里面定义了关于 graph 的一些基本操作
    # get_default_graph 返回默认的 graph
    self._graph = ops.get_default_graph()
      
  @property
  def graph(self):
    return self._graph
  ```

  在初始化 Session 时若没有指定 graph, 那么由于继承了 BaseSession 类，就会执行 ops.get_default_graph() 获取默认的 graph 做为自己的 graph。

  ```python
  from tensorflow.python.framework import ops
  sess = tf.Session()
  sess_ = tf.Session()
  sess.graph == sess_.graph == ops.get_default_graph()  # 或者直接用 tf.get_default_graph()
  # True
  ```

**方法**：

- `__init__(target='', graph=None, config=None)`

  初始化 target 指定了 TensorFlow 的执行引擎，和分布式有关。

  config 用一个 ConfigProto 来设置配置。

- `__enter__() 和 __exit__()`

  用 with 语句创建上下文管理器时执行的操作，从上面源码可看到，`__enter__` 主要目的是把当前 Session 设置为默认 Session，然后返回一个上下文管理器使之生效。

- as_default()

  ```python
  # 这一部分定义在 class BaseSession 中
  def as_default(self):
    return ops.default_session()
  ```

  返回一个上下文管理器，把当前 session 设定为默认 Session。注意，必须在 with 语句中定义才能生效。在默认 Session 中，就可以直接使用 tf.Operation.run() 函数和 tf.Tensor.eval() 函数了。

  ```python
  c = tf.constant([1])
  sess = tf.Session()
  with sess.as_default():
    assert tf.get_default_session() is sess
    print c.eval()
   
  # 上面还可以写成:
  # with sess:
  #	pass
  # 或者干脆：
  # with tf.Session() as sess:
  #	pass
  ```

- close()

  关闭 Session，释放资源

- list_devices()

  以 list 形式返回可用的设备。

  ```python
  print sess.list_devices()[0]
  # _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 140229230322672)
  print sess.list_devices()[0].name
  # /job:localhost/replica:0/task:0/device:CPU:0
  print sess.list_devices()[0].device_type
  # CPU
  ```

- run(fetches, feed_dict=None, options=None, run_metadata=None)

  执行 operations 并计算 fetches 中 tensor 的值。fetches 可以为单个 graph 元素，或者用 list, tuple, namedtuple, dict, OrderedDict 包起来的 graph 元素。所谓的 graph 元素，包括: tf.Operation, tf.Tensor, tf.SparseTensor, get_tensor_handle 和 string (graph 中 tensor 或者 op 的名字)

## 2. InteractiveSession

tf.InteractiveSession() 与 tf.Session() 的区别是: 前者一旦建立了，就成了默认的 Session，就可以直接调用 Tensor.eval() 和 Operation.run() 操作了。从 InteractiveSession 源码看，也很好理解：

```python
class InteractiveSession(BaseSession):

    if not config:
      gpu_options = config_pb2.GPUOptions(allow_growth=True)
      config = config_pb2.ConfigProto(gpu_options=gpu_options)
    config.graph_options.place_pruned_graph = True

    super(InteractiveSession, self).__init__(target, graph, config)
    self._default_session = self.as_default()
    self._default_session.enforce_nesting = False
    self._default_session.__enter__()
    self._explicit_graph = graph
    if self._explicit_graph is not None:
      self._default_graph = graph.as_default()
      self._default_graph.enforce_nesting = False
      self._default_graph.__enter__()

  def close(self):
    super(InteractiveSession, self).close()
    if self._explicit_graph is not None:
      self._default_graph.__exit__(None, None, None)
    self._default_session.__exit__(None, None, None)
```

`self._default_session = self.as_default()` 将自己设定为默认 Session，`self._default_session.__enter__()` 返回上下文管理器。因此在交互式终端中使用就很方便了。



