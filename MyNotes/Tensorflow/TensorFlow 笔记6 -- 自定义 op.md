自定义 op

为了提高效率，Tensorflow中的操作内核完全是用C ++编写，但是在C ++中编写Tensorflow内核可能会相当痛苦。通过 `tf.py_func(func, inp, Tout, stateful=True, name=None)` 可以将任意的 python 函数 `func`转变为 TensorFlow op。

然而这样的缺点也很明显：效率不够高，因为这样写出的 python 代码是无法在 GPU 上加速跑的。因此常用于作原型设计，一旦 idea 验证通过了，就应该去用 C++ 从底层内核实现。

参数说明: 

- func: 目标函数
- inp: 输入变量，类型为 numpy array；有多个输入值时，用 list 包起来
- Tout: 输出的数据类型

举个例子：

```python
sess = tf.Session()

def new_relu(inputs):
	
	def _relu(x):
		return np.maximum(x, 2.0)
	
	output = tf.py_func(_relu, [inputs], tf.float32)
	
	return output

x = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)

y = new_relu(x)

print sess.run(y)
# [ 2.  2.  3.  4.  5.]
```

