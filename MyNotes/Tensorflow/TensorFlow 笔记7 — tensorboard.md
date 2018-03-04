# TensorFlow 笔记6 — tensorboard

tf.summary.scalar(name, tensor, collections=None): 记录标量信息。默认保存到`[GraphKeys.SUMMARIES]`这个collection. 返回一个 Summary protobuf。

tf.summary.histogram(name, tensor, collections=None): 记录数据标量。返回一个 Summary protobuf。

TensorFlow 中的 op 除非有依赖关系，否则是不会执行的。因此一般用 `tf.summary.merge_all()`把以上所有 op 组合成一个 op，然后执行。执行此组合 op 后，会输出一个序列化的 Summary protobuf 文件，用tf.summary.FileWriter() 函数保存到硬盘。

tf.summary.FileWriter(logdir, graph=None, flush_secs=120): logdir 为储存路径，传入 graph后才会在 tensorboard 中显示图的结构，flush_secs 指定多少秒后写入硬盘。

例子：同时记录训练集和测试集的 acc:

```python
with tf.Session() as sess:
    train_step = tf.train.AdamOptimizer(1e-4).minimize(losses)
    accuracy = ...
    tf.summary.scalar('acc', accuracy)
    
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train/', sess.graph)
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test/')
    
    for i in range(2000):
        summary, _ = sess.run([merged, train_step], feed_dict={(train_data here)})
        train_writer.add_summary(summary, i)
    	
        if i % 10 == 0 and i > 0:
            summary, test_acc = sess.run([merged, accuracy], feed_dict={(test_data here)})
            test_writer.add_summary(summary)
    
    train_writer.close()
    test_writer.close()
```

这样在 LOG_DIR 下会有 train 和 test 两个文件夹，使用时在终端中使用 tensorboard —logdir=LOG_DIR 就可以了。

如果只想记录 accuracy 这一个值，不想全部 merge 成一个 op，可以使用 tf.summary.merge() 函数。

tf.summary.merge(inputs, collections=None, name=None): inputs 为 list，每个元素是序列化后的 Summary protobuf 文件。

改写上面例子，只记录测试集精度：

```python
with tf.Session() as sess:
    train_step = tf.train.AdamOptimizer(1e-4).minimize(losses)
    accuracy = ...
    tf.summary.scalar('acc', accuracy)
    
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge([accuracy])  # 只记录accuracy
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test/')
    
    for i in range(2000):
        sess.run(train_step, feed_dict={(train_data here)})
    	
        if i % 10 == 0 and i > 0:
            summary, test_acc = sess.run([merged, accuracy], feed_dict={(test_data here)})
            test_writer.add_summary(summary)
```

- 技巧: 如何分 batch 训练的时候，如何计算整个 epoch 平均后的 acc 然后记录到 Summary protobuf中？

方法一: 使用 tf.metrics.accuracy() 函数或者 tf.contrib.metrics.streaming_mean() （已废弃）

```python
accuracy = ...
streaming_accuracy, streaming_accuracy_update = tf.metrics.accuracy(accuracy)
streaming_summary = tf.summary.scalar('streaming_accuracy', streaming_accuracy_update)

merged = tf.summary.merge_all()

for i in range(epoch_num):
    sess.run(tf.local_variables_initializer()) # 因为 tf.metrics.accuracy 引入了局部变量，在一个batch内累计求平均值，所以每执行一个batch要置0
    for j in range(batch_num):
        summ = sess.run(merged, feed_dict={})
        writer.add_summary(summ)
```

方法二：直接手动每个batch计算完acc后，再操作：

如果是在图之外用np计算得到数值后，需要自定义summary protobuf，输入进去：

```python
for batch in validation_set:
	accuracies.append(sess.run([training_op]))
accuracy = np.mean(accuracies)

# 自定义 Summary protobuf 文件:
from tensorflow.core.framework import summary_pb2
def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, 
                                                                simple_value=val)])
summary_writer.add_summary(make_summary('Accuracy', accuracy), global_step)

# 上面也可以简写为
summary = tf.Summary()
summary.value.add(tag="Accuracy", simple_value=accuracy)
summary_writer.add_summary(summary, global_step)
```

如果是在图内计算batch内累计平均值，可以参考tf.metrics.accuracy源码设定local_variables: count 和 count 来计算，但记得运行每个batch前要清0。或者更简单的，用滑动平均来搞定：(其实这样就有点偏差了)

```python
accuracy = ...
ema = tf.train.ExponentialMovingAverage(decay=0.9)
maintain_ema_op = ema.apply(accuracy)

with tf.control_dependencies([train_step]):
    train_op = tf.group(maintain_ema_op)
    
sess.run(train_op)
acc = sess.run(ema.average(accuracy)
summary_writer.add_summary('Accuracy', acc)
```

