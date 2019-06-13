---
layout: post
title: mxnet、pytorch、tensorflow和keras的区别
categories: [深度学习]
description: 总结mxnet、pytorch、tensorflow、keras的一些区别。
keywords: 深度学习, mxnet, pytorch, tensorflow, keras
---

模型开发需要对数据的预处理、数据增强、模型定义、权值初始化、模型 Finetune、学习率调整策略、损失函数选取、优化器选取、可视化等等。

[Pytorch官网](https://pytorch.org/docs/stable/autograd.html#variable-deprecated)

[pytorch学习笔记](https://blog.csdn.net/JackyTintin/article/details/70233044)

[如何从TensorFlow迁移到PyTorch（没仔细看）](https://www.jianshu.com/p/c1d9cdb52548)

[MXNet/Gluon 中网络和参数的存取方式（没看）](https://blog.csdn.net/caroline_wendy/article/details/80494120)

## Variable和Tensor

[Pytorch  Variable API has been deprecated](https://discuss.pytorch.org/t/what-is-the-difference-between-tensors-and-variables-in-pytorch/4914/6)

## 自动求导

```
[optimizer.step()可以重复执行多次吗？](https://www.baidu.com/s?wd=optimizer.step()%E5%8F%AF%E4%BB%A5%E9%87%8D%E5%A4%8D%E6%89%A7%E8%A1%8C%E5%A4%9A%E6%AC%A1%E5%90%97&rsv_spt=1&rsv_iqid=0x8212103e000034c6&issp=1&f=8&rsv_bp=0&rsv_idx=2&ie=utf-8&tn=baiduhome_pg&rsv_enter=1&rsv_sug3=31&rsv_t=7019Hp%2Fmx%2F7DdhZ7JCS90pYhCo4xV5mpq5eM0yy4VPvCrCI3wOPwZ3VlBrd5bylWzhDl&rsv_sug1=4&rsv_sug7=100&rsv_sug2=0&inputT=7066&rsv_sug4=7066)
```

mxnet若要对一个变量x求导(x = nd.arange(4).reshape((4, 1)))，首先x.attach\_grad()申请存储梯度所需要的内存。为了减少计算和内存开销，默认条件下MXNet**不会记录用于求梯度的计算**，调用autograd.record()来要求MXNet记录与求梯度有关的计算。接下来我们可以通过调用backward()函数自动求梯度。

pytorch：

```
optimizer.zero_grad()   # 清空上一步的残余更新参数值
loss.backward()         # 误差反向传播, 计算参数更新值
optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
```

mxnet每次计算完会清空上一次的梯度更新值，pytorch不会清空，所以需要执行zaro grad操作。

**pytorch如何截断梯度传播？**
 
1.使用detach和detach\_函数

如果我们有两个网络A、B，两个关系是这样的y=A(x)、z=B(y)现在我们想用z.backward()来为B网络的参数来求梯度，但是又不想求A网络参数的梯度。我们可以这样：

```
# y=A(x), z=B(y) 求B中参数的梯度，不求A中参数的梯度

# 第一种方法
y = A(x)
z = B(y.detach())
z.backward()

# 第二种方法
y = A(x)
y.detach_()
z = B(y)
z.backward()
```

在这种情况下，detach和detach\_都可以用。但是如果你也想用y来对A进行BP呢？那就只能用第一种方法了。**因为第二种方法已经将A模型的输出给detach（分离）了**。

2.例如a是一个tensor，a是A网络输出，现在a又是B网络的输入，B网络输出为b；即a=A(i)、b=B(a)；如果我们对b\.backward()，那么A网络也更新了，我们只想更新B网络，然后随后更新A网络。可以这样操作

```
a=A(i)
a1=Variable(a)
b=B(a)
b.backward()
这样只会更新B网络，因为B网络输入是用A网络输出a初始化的变量a1，a1是叶子结点。而且重要的是变量a1也保存了B网络反向的梯度，我们可以用变量a1保存的梯度来更新A网络。
这种情况适用于A、B网络更新策略不同。
```

[自动求导机制](https://pytorch-cn.readthedocs.io/zh/latest/notes/autograd/)

3.利用model.parameters()，修改requires\_grad 属性。

keras是高层api，反向传播不需要backward和step。

tensorflow：**可以通过stop gradient阻止部分该部分的反向传播；可以通过apply gradient来阻止部分变量更新，compute gradient计算出来梯度之后，只对需要更新的变量执行apply gradient。**

* opt\.compute\_gradients()
* opt\.apply\_gradients()
* tf\.gradients()：计算梯度的函数。
* tf\.stop\_gradient()：阻挡节点BP的梯度。

[（注意可以用tf gradients求导的一定是浮点数，整数不可以）（重要）tensorflow学习笔记（三十）：tf.gradients 与 tf.stop\_gradient() 与 高阶导数](https://blog.csdn.net/u012436149/article/details/53905797)

[opt.compute\_gradients() 与 tf.gradients 与 tf.stop\_gradient()](https://blog.csdn.net/u012871493/article/details/71841709)


```
a = tf.multiply(w1, 3.0)
a_stoped = tf.stop_gradient(a)
下一步用a_stoped参与计算，a_stoped之前的Variable梯度计算都受影响，如果Variable梯度只能来自于a，那么梯度计算结果为None。
```

[TensorFlow学习笔记（一）](https://blog.csdn.net/u013745804/article/details/78332151)

tensorflow中有一个计算梯度的函数tf.gradients(ys, xs)，要注意的是，xs中的x必须要与ys相关，不相关的话，会报错。tf.gradients()中的要求grad\_ys与ys维度一致的要求。

在使用tf.gradients()时，我有一个疑惑，**返回值sum(dy/dx)到底是对多个输出变量进行求和，还是对batch进行求和，还是两者都需要求和呢**？

如果，仅仅输入一个sample（也就是说不考虑batch），那么我们对于参数的更新，应该是要对多个输出变量对于该参数的导数进行求和的。那么该函数是否会对batch进行求均值处理？答案是**会对batch进行求和，但不是求均值**。我们可以看出，最终的返回值将**会对各个batch以及各个输出变量关于参数的导数进行求和**。

## 模型参数初始化

mxnet定义了模型net = nn\.Sequential()，然后调用net\.initialize()来初始化模型参数。

pytorch模型定义时就顺便初始化了，不需要专门的初始化步骤。

```
https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
定义Linear层参数初始化过程。
```

tensorflow需要在session中执行参数初始化的操作。

keras高阶api。

## 模型的训练和测试

pytorch在模型训练时会在前面加上**model.train()**，在测试模型时在前面使用**model.eval()**。当然如果你不写这两句程序也可以运行，这是因为这两个方法针对在网络训练和测试时采用不同方式的情况，比如Batch Normalization和Dropout。

[PyTorch(七)——模型的训练和测试、保存和加载](https://blog.csdn.net/u011276025/article/details/78507950)

mxnet的mod.forward()里面有一个参数is\_train，**训练的时候设成True，测试的时候设成False就成了**。

tensorflow中的dropout和batch norm类似的层都有一个参数叫做is\_training。所以设置一个is\_training的placeholder，然后训练和测试阶段分别传入不同的值。如下

```
is_training = tf.placeholder(tf.bool, name='is_training')

summary_str, _, loss_, acc_ = sess.run([merged_summary_op, train_step, loss, accuracy], feed_dict={inputs: images_, labels: truths_, is_training: True})

valid_str, vloss, vacc = sess.run([merged_summary_op, loss, accuracy], feed_dict={inputs: valid_imgs_, labels: valid_trus_, is_training: False})
```

keras的fit函数可以指定x、y和validation\_data，感觉内部其实都已经设好了。keras的api太高阶了。

## 数据读取

PyTorch中数据读取的一个重要接口是torch.utils.data.DataLoader，该接口定义在dataloader.py脚本中，只要是用PyTorch来训练模型基本都会用到该接口，该接口主要用来将**自定义的数据读取接口的输出**或者**PyTorch已有的数据读取接口的输入**按照batch size**封装成Tensor**，后续只需要再包装成Variable即可作为模型的输入，因此该接口有点承上启下的作用，比较重要。

[PyTorch源码解读之torch.utils.data.DataLoader](https://blog.csdn.net/u014380165/article/details/79058479)

[pytorch实现自由的数据读取－torch.utils.data的学习](https://blog.csdn.net/tsq292978891/article/details/79414512)

## 维度增加和维度交换以及维度扩展

pytorch中的view和permute函数。

view之前最好用contiguous函数把tensor移到连续的地址上。

[PyTorch中view的用法](https://blog.csdn.net/york1996/article/details/81949843)

[PyTorch中permute的用法](https://blog.csdn.net/york1996/article/details/81876886)

expand：维度扩展，扩展某个size为1的维度，例如(2,2,1)扩展为(2,2,3)。**expand没有分配内存**，所以改原来维度为1位置的数据，整行数据全改。

expand\_as：等于expand(tensor.size())。

repeat：沿着指定的维度重复张量。**不同于expand()方法，本函数复制的是张量中的数据**。

[PyTorch 常用方法总结4：张量维度操作（拼接、维度扩展、压缩、转置、重复……）](https://zhuanlan.zhihu.com/p/31495102)

## 模型定义

模型定义很多时候需要调用已有的接口，例如tensorflow的tf\.nn\.conv2d和pytorch的torch\.nn\.Conv2d接口，这两个接口第一个返回**tensor**，第二个返回Conv2d类型。因为tensorflow的conv2d接口定义的是计算过程；而pytorch的conv2d接口定义的是参数，计算过程在forward函数中定义。当然这也是静态图计算和动态图计算的最大区别了，**静态图直接建立图把计算过程都定义好了**。

对与tensorflow来说，它本身各种运算都在一个默认计算图中，一般tennsorflow也只定义一个默认图。

[tensorflow之自定义神经网络层](https://blog.csdn.net/u013230189/article/details/81742306)

[ensorflow：自定义op简单介绍](https://blog.csdn.net/u012436149/article/details/73737299)

层继承tf.keras.layers.Layer和网络继承tf.keras.Model)。

[PyTorch 基礎篇](https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305d9cd231015d9d0992ef0030)

pytorch：

定义model：继承自torch.nn.Module；\_\_init\_\_函数先初始化父类super().\_\_init\_\_()，然后定义各个层（其实就是定义参数）；forward函数定义了这个model的层之间的运算关系，这个函数的输入是self和x，x是这个子模块的输入。

定义网络层：继承自Function；forward()是网络层的计算操作，能够根据你的需要设置参数；backward()是梯度计算操作。

[pytorch定义网络](https://zhuanlan.zhihu.com/p/39681524)

[Pytorch： 自定义网络层](https://blog.csdn.net/xholes/article/details/81478670)

[Pytorch：自定义网络层](https://blog.csdn.net/cham_3/article/details/79607916)

[Pytorch入门学习（八）-----自定义层的实现（甚至不可导operation的backward写法）（非常全）](https://blog.csdn.net/Hungryof/article/details/78346304)

[扩展PyTorch（讲的不错）](https://pytorch-cn.readthedocs.io/zh/latest/notes/extending/#pytorch)

对于前馈传播网络，如果每次都写复杂的forward函数会有些麻烦，有两种简化方式：ModuleList、Sequential。

Keras:

[Keras官方中文版文档正式发布了](https://baijiahao.baidu.com/s?id=1594183106083021263&wfr=spider&for=pc)

Sequential模型、函数式API（Model）。

**Sequential模型**：

[快速开始Sequential模型](https://keras-cn.readthedocs.io/en/latest/legacy/getting_started/sequential_model/#merge)

**多个Sequential模型可经由一个Merge层合并到一个输出**。Merge层的输出是一个可以被添加到新Sequential的层对。下面这个例子**将两个Sequential合并到一起**：

```
https://github.com/skckompella/memn2n-keras/blob/master/src/single_layer.py 这里也使用了Merge合并多个Sequential模型。

from keras.layers import Merge

left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784))

right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784))

merged = Merge([left_branch, right_branch], mode='concat')

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(10, activation='softmax'))

这个两个分支的模型可以通过下面的代码训练:
final_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
final_model.fit([input_data_1, input_data_2], targets)
```

**Model模型**：

[快速开始泛型模型](https://keras-cn.readthedocs.io/en/latest/legacy/getting_started/functional_API/)

**所有的模型都是可调用的，就像层一样**。利用泛型模型的接口，我们可以很容易的**重用已经训练好的模型**：你可以把模型当作一个层一样，通过提供一个tensor来调用它。注意当你调用一个模型时，你**不仅仅重用了它的结构，也重用了它的权重**。

如下图，model和model1的的模型参数是一样的。

```
https://github.com/alecGraves/cyclegan_keras/blob/master/test/mnist_test.py 这里也是把Model当作层来调用的例子，然后调用之后定义一个新的模型，新的模型参数也包括被调用Model的参数。

from keras.layers import Input, Dense
from keras.models import Model
inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(input=inputs, output=predictions)

x = Input(shape=(784,))
y = model(x)
model1 = Model(input=x, output=y)
model1.summary()
```

[感觉有问题 使用Keras编写自定义网络层（layer）](https://blog.csdn.net/u013084616/article/details/79295857)

[Keras Layer自定义](https://www.jianshu.com/p/6c34045216fb)

自定义网络层：

1.keras.core.lambda()。
2.Layer继承类。build(input\_shape)：这是定义权重的方法，这个方法必须设置self.built = True，可通过调用super(layer,self).build()实现。call(self, x, mask=None)是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心call的第一个参数。

## 获取模型中的变量，输出某一个tensor的值

tensorflow：

获取模型中的变量：

[tf.trainable\_variables和tf.all\_variables的对比](https://blog.csdn.net/UESTC_C2_403/article/details/72356448)

[TensorFlow教程·模型参数保存和提取](https://zhuanlan.zhihu.com/p/40523154)

1.tf.trainable\_variables()生成一个list，里面是所有可训练的变量；tf.all\_variables()生成一个list，里面是所有变量。其中类型是Variable，通过\.name可以得到变量的名字。
2.tf.train.NewCheckpointReader()

[TensorFlow教程·模型参数保存和提取](https://zhuanlan.zhihu.com/p/40523154)

输出某一个tensor的值：

1.如果是tensor在图中有定义，直接用sess.run输出。
2.**如果是导入的模型，通过tensor的名字输出**。

```
tf.get_default_graph().get_tensor_by_name('MNIST/f2_layer/add:0')
```

3.tf.train.NewCheckpointReader()类。

```
reader = tf.train.NewCheckpointReader('./model.ckpt')
w = reader.get_tensor("Variable_2")
print(type(w))
print(w.shape)
print(w[0])
```

pytorch：

[PyTorch 第四弹\_通过LeNet初识pytorch神经网络\_上](https://www.cnblogs.com/hellcat/p/6858125.html)

[【pytorch】Module.parameters()函数实现与网络参数管理](https://blog.csdn.net/idwtwt/article/details/82195000)

[How to print model’s parameters with its name and \`requires\_grad value\`?](https://discuss.pytorch.org/t/how-to-print-models-parameters-with-its-name-and-requires-grad-value/10778)

1.module\.parameters()，返回参数值。返回全部的参数值，迭代器。
2.module\.named\_parameters()，返回（参数名称，参数值）。返回参数名称和值，迭代器。

keras：

有一篇自己写的Keras文档中有相关内容。

## 模型中参数定义

pytorch:

torch\.nn\.Parameter来定义Model的参数。

[pytorch：在网络中添加可训练参数，修改预训练权重文件](https://blog.csdn.net/qq_19672579/article/details/79195568)

## 模型持久化和加载

tensorflow：

[tensorflow保存加载模型查看训练参数](https://blog.csdn.net/James_Ying/article/details/70224680)

```
持久化：
saver = tf.train.Saver()
saver.save(sess, '')
读取：
saver = tf.train.import_meta_graph()
with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    
    x = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
    y_ = tf.get_default_graph().get_tensor_by_name('Placeholder_1:0')
    keep_prob = tf.get_default_graph().get_tensor_by_name('Placeholder_2:0')

    result = sess.run(tensor_temp, feed_dict = {x: batch[0], y_: batch[1], keep_pr
ob: 0.5})
```

Keras:

Pytorch：

[Pytorch自由载入部分模型参数并冻结](https://zhuanlan.zhihu.com/p/34147880)

[保存提取](https://morvanzhou.github.io/tutorials/machine-learning/torch/3-04-save-reload/)

[PyTorch 学习笔记（五）：存储和恢复模型并查看参数](https://www.pytorchtutorial.com/pytorch-note5-save-and-restore-models/)

[tate\_dict](https://pytorch-cn.readthedocs.io/zh/latest/package\_references/torch-nn/)

```
torch.save(net1, 'net.pkl')  # 保存整个网络
torch.save(net1.state_dict(), 'net_params.pkl')   # 只保存网络中的参数 (速度快, 占内存少)

提取整个神经网络：
net2 = torch.load('net.pkl')
提取参数：
net3.load_state_dict(torch.load('net_params.pkl'))
```

## 只训练部分参数、加载部分参数

tensorflow：

[tensorflow 固定部分参数训练，只训练部分参数](https://blog.csdn.net/guotong1988/article/details/84562172)

为self\.optim\.minimize函数传入var\_list参数。

[tensorflow加载部分层方法](https://blog.csdn.net/jinglingli_sjtu/article/details/69950491)

选择在加载网络之后重新初始化剩下的层。

[tensorflow 恢复部分参数、加载指定参数](https://blog.csdn.net/b876144622/article/details/79962727)

saver = tf\.train\.Saver(variables\_to\_restore)传入需要恢复的参数列表var\_list。保存的时候也可以设置这个变量指定要保存的参数列表。

pytorch：

[PyTorch参数初始化和Finetune](https://zhuanlan.zhihu.com/p/25983105)

局部微调：有时候我们加载了训练模型后，**只想调节最后的几层，其他层不训练**。其实不训练也就意味着不进行梯度计算，PyTorch中提供的requires\_grad使得对训练的控制变得非常简单。

全局微调：有时候我们需要**对全局都进行finetune**，只不过我们希望**改换过的层和其他层的学习速率不一样**，这时候我们可以把其他层和新层在optimizer中单独赋予不同的学习速率。

[Pytorch - 网络模型参数初始化与 Finetune](https://www.aiuai.cn/aifarm615.html)

**修改模型内部网络层**：如果需要对网络的内部结构进行改动，则需要采用参数覆盖的方法。即，**先定义类似网络结构，再提取预训练模型的权重参数，覆盖到自定义网络结构中**。

model\.state\_dict()：是一个简单的python的字典对象，将每一层与它的对应参数建立映射关系。

model\.parameters()输出网络所有的Parameter（有值和require\_grad属性）。

## 输出中间层结果

[keras输出中间层结果的2种方法](https://blog.csdn.net/hahajinbu/article/details/77982721)

pytorch:

[Extracting and using features from a pretrained model](https://discuss.pytorch.org/t/extracting-and-using-features-from-a-pretrained-model/20723)

[How to extract features of an image from a trained model](https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3)

## reshape

keras: layers\.Reshape

tensorflow: tf\.reshape

tf\.reshape和keras\.layers\.Reshape不一样，**tf针对所有的维度进行reshape，keras只针对0维以外的进行reshape**。

[keras.layers.Reshape和tf.reshape比较](https://blog.csdn.net/weixin_43331915/article/details/83305263)

## 输入

keras: keras\.layers\.Input 例如keras\.layers\.Input(shape=(None, None, None, 36))表示五个维度

tensorflow: tf.placeholder 例如tf\.placeholder(tf\.int32, \[None, 36\])表示两个纬度

## 矩阵乘法

keras: tf\.keras\.backend\.batch\_dot: 批量化的点积。K\.dot

[参考1](https://www.w3cschool.cn/tensorflow_python/tf_keras_backend_batch_dot.html)

[参考2](https://blog.csdn.net/weixin_34389926/article/details/88261323)

tensorflow: tf\.matmul、tf\.batch_matmul、tf\.multiply

batch_dot和matmul差不多。K\.dot会沿着两个矩阵最后两个维度进行乘法，不是element-wise矩阵乘法。

[理解keras中的batch_dot，dot方法和TensorFlow的matmul](https://blog.csdn.net/huml126/article/details/88739846)

新版的tensorflow已经移除batch\_matmul，使用时换为matmul就可以了。

[TensorFlow中的tf.batch_matmul()](https://blog.csdn.net/yyhhlancelot/article/details/81191923)

[TensorFlow中如何实现batch_matmul](https://www.jianshu.com/p/afe96784cbbf)

**重点**：感觉batch\_dot只能计算只能涉及最后两个维度。

[Understanding batch_dot() in Keras with Tensorflow backend](https://stackoverflow.com/questions/54057742/understanding-batch-dot-in-keras-with-tensorflow-backend)

tf\.multiply: 矩阵x和矩阵y对应位置的元素相乘。

