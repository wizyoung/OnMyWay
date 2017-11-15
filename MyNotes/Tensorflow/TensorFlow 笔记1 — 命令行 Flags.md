# TensorFlow 笔记1 — 命令行 Flags

看例子：

```python
# coding: utf-8
import tensorflow as tf

flags = tf.flags
# flags = tf.app.flags # tf.app.flags 本质也是调用 tf.flags, 二者是一样的

flags.DEFINE_integer('epoch', 25, 'Epoch to train [25]')
flags.DEFINE_float('learning_rate', 0.002, 'Learning rate for adam [0.002]')

FLAGS = flags.FLAGS  # 通过 FLAGS 创建一个_FlagValues对象

# 第一次调用 FLAGS.epoch 时, epoch 不存在，因此调用源码中 _FlagValues 类的__getattr__函数，这时
# parsed 是 false, 因此调用 _parse_flags() 对参数进行解析
print FLAGS.epoch
print FLAGS.learning_rate

print FLAGS.__dict__['__flags']  # 返回解析后的所有的 flags 的键值对
```

输出:

```
25
0.002
{'epoch': 25, 'learning_rate': 0.002}
```

使用方法：

(1) `flags = tf.flags` 或者 `flags = tf.app.flags`

两者一样，都是调用 `tensorflow.python.platform.flags`。

(2) 定义参数:

`flags.DEFINE_integer(flag_name, default_value, docstring)` 形式定义参数.

可用的参数定义有(DEFINE_后的): bool, boolean (和 bool 等价), float, integer, string.

(3) `FLAGS = flags.FLAGS` 创建一个 FLAGS (_FlagValues) 对象，便于对参数进行解析。

终端运行时的用法:

```
python test.py --epoch 20 --learning_rate 0.002
```

-------------

源码解析：

TensorFlow 中的 Flags 其实就是一个ArgumentParser的简单包装，源码在 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/platform/flags.py

核心代码如下:

```python
# flags.py

# 创建 ArgumentParser
import argparse as _argparse
_global_parser = _argparse.ArgumentParser()  

# 定义 _FlagValues 类
class _FlagValues(object):

  def __init__(self):
    self.__dict__['__flags'] = {}
    self.__dict__['__parsed'] = False
    self.__dict__['__required_flags'] = set()

  # 解析参数，并将参数键值对加到字典中
  def _parse_flags(self, args=None):
    result, unparsed = _global_parser.parse_known_args(args=args)
    for flag_name, val in vars(result).items():
      self.__dict__['__flags'][flag_name] = val
    self.__dict__['__parsed'] = True
    self._assert_all_required()
    return unparsed

  def __getattr__(self, name):
    """Retrieves the 'value' attribute of the flag --name."""
    try:
      parsed = self.__dict__['__parsed']
    except KeyError:
      # May happen during pickle.load or copy.copy
      raise AttributeError(name)
    if not parsed:
      self._parse_flags()
    if name not in self.__dict__['__flags']:
      raise AttributeError(name)
    return self.__dict__['__flags'][name]

  def __setattr__(self, name, value):
    """Sets the 'value' attribute of the flag --name."""
    if not self.__dict__['__parsed']:
      self._parse_flags()
    self.__dict__['__flags'][name] = value
    self._assert_required(name)

  def _add_required_flag(self, item):
    self.__dict__['__required_flags'].add(item)

  def _assert_required(self, flag_name):
    if (flag_name not in self.__dict__['__flags'] or
        self.__dict__['__flags'][flag_name] is None):
      raise AttributeError('Flag --%s must be specified.' % flag_name)

  def _assert_all_required(self):
    for flag_name in self.__dict__['__required_flags']:
      self._assert_required(flag_name)
      
# 提供一个全局标志，方便调用
FLAGS = _FlagValues()

# 利用 argparse 的 add_argument()函数添加参数
def _define_helper(flag_name, default_value, docstring, flagtype):
  """Registers 'flag_name' with 'default_value' and 'docstring'."""
  # 这里 flag_name 前默认加上了‘--’, 因此是可选参数
  _global_parser.add_argument('--' + flag_name,
                              default=default_value,
                              help=docstring,
                              type=flagtype)

# DEFINE_string 是用上面的_define_helper 实现的
def DEFINE_string(flag_name, default_value, docstring):
  _define_helper(flag_name, default_value, docstring, str)
```

可以看到，首先导入 flags.app 时，ArgumentParser()已经建立好，然后用 DEFINE_xxx 函数先预定义一些变量，再用 FLAGS = flags.FLAGS 创建 FLAGS 对象，也就是 _FlagValues() 对象。

那么在用 flags.xxx 访问变量 xxx 时，显然 xxx 是不存在的，就会调用_FlagValues 类的 `__getattr__()` 函数，进而调用 `_parse_flags()` 函数进行参数解析。


