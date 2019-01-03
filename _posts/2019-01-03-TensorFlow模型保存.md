---
layout: post
title: TensorFlow模型保存
categories: [TensorFlow]
description: some word here
keywords: TensorFlow, 模型保存
---

## TensorFlow的模型格式

1.**CheckPoint（\.ckpt）**

在训练TensorFlow模型时，每迭代若干轮需要保存一次权值到磁盘，称为“checkpoint”。这种格式文件是由tf\.train\.Saver()对象调用saver\.save()生成的，只包含若干Variables对象序列化后的数据，**不包含图结构**，所以只给checkpoint模型不提供代码是无法重新构建计算图的。载入checkpoint时，调用saver\.restore(session, checkpoint\_path)。

2.**MetaGraphDef（\.meta）**

类：MetaGraphDef，包含MetaInfoDef、GraphDef、SaverDef、CollectionDef。

序列化存储格式：protobuf，\.meta文件。

3.**GraphDef（\.pb）**

这种格式文件包含protobuf对象序列化后的数据，**包含了计算图**，可以从中得到所有运算符（operators）的细节，也包含张量（tensors）和Variables定义，**但不包含Variable的值**，因此**只能从中恢复计算图，但一些训练的权值仍需要从checkpoint中恢复**。

利用\.pb文件构建计算图：

```
def load_graph(model_file):
	graph = tf.Graph()
	graph_def = tf.GraphDef()
	
	with open(model_file, "rb") as f:
		graph_def.ParseFromString(f.read())
	with graph.as_default():
		tf.import_graph_def(graph_def)

	return graph
```

4.**FrozenGraphDef（\.pb）**

TensorFlow一些例程中用到\.pb文件作为预训练模型，这**和上面GraphDef格式稍有不同**，属于**冻结（Frozen）后的GraphDef文件**，简称FrozenGraphDef格式。这种文件格式**不包含Variables节点**。将GraphDef中所有**Variable节点转换为常量**（其值从checkpoint获取），就变为**FrozenGraphDef格式**。代码可以参考 tensorflow/python/tools/freeze\_graph.py。\.pb 为**二进制文件**，实际上protobuf也支持文本格式（\.pbtxt），但包含权值时文本格式会占用大量磁盘空间，一般不用。

## TensorFlow固化模型

[TensorFlow固化模型](https://www.jianshu.com/p/091415b114e2)

[扣丁学堂浅谈将TensorFlow的模型网络导出为单个文件的方法（讲的很好）](https://www.codingke.com/article/2915)

[Tensorflow 模型文件格式转换（重点是\.pb和\.pbtxt的转换）](https://blog.csdn.net/jinying2224/article/details/78037926)

1.利用TensorFlow提供的接口freeze\_graph\.py。

2.convert\_variables\_to\_constants

其实在TensorFlow中传统的保存模型方式是保存常量以及graph的，而我们的权重主要是变量，**如果我们把训练好的权重变成常量之后再保存成PB文件，这样确实可以保存权重**，就是方法有点繁琐，需要一个一个调用eval方法获取值之后赋值，再构建一个graph，把W和b赋值给新的graph。

```
Google编写了一个方法供我们快速的转换并保存。

from tensorflow.python.framework.graph_util import convert_variables_to_constants

在想要保存的地方加入如下代码，把变量转换成常量。这里参数第一个是当前的session，第二个为graph，第三个是输出节点名。
output_graph_def = convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output/predict'])

生成文件。
with tf.gfile.FastGFile('model/CTNModel.pb', mode='wb') as f:
	f.write(output_graph_def.SerializeToString())
```

测试保存的模型正常读取、运行。

```
newInput_X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH], name="X")

with open('./model/rounded_graph.pb', 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	output = tf.import_graph_def(graph_def,
		input_map={'inputs/X:0': newInput_X},
		return_elements=['output/predict:0'])

text_list = sesss.run(output, feed_dict={newInput_X: [captcha_image]})
```

## 参考

[TensorFlow 到底有几种模型格式？（SavedModel接口没看）](https://cloud.tencent.com/developer/article/1009979)

[tensorflow中的几种模型文件](tensorflow中的几种模型文件)

[tensorflow，使用freeze\_graph\.py将模型文件和权重数据整合在一起并去除无关的Op（利用tensorflow提供的接口）](https://blog.csdn.net/czq7511/article/details/72452985)

