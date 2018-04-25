---
layout: post
title: RNN GRU LSTM
categories: [深度学习, NLP]
description: 介绍在NLP中广泛使用的一些神经网络。
keywords: NLP, 神经网络, 机器学习
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## RNN

### 为什么要有RNN?

当人们在思考的时候，不会从头开始，而是保留之前思考的一些结果来为现在的决策提供支持。同时，我们会根据上下文的信息来理解一句话的含义。**但是传统的神经网络并不能实现这个功能，这就是其一大缺陷。**而RNN的结构的最大特点是神经元的某些输出可作为其输入再次传输到神经元中，因此可以利用之前的信息。

DNN和CNN中，训练样本的输入和输出是比较的确定的。但是有一类问题DNN和CNN不好解决，就是训练样本输入是连续的序列,且序列的长短不一，比如基于时间的序列：一段段连续的语音，一段段连续的手写文字。这些序列比较长，且长度不一，比较难直接的拆分成一个个独立的样本来通过DNN/CNN进行训练。而对于这类问题，RNN比较擅长。

### RNN的网络结构

![](/images/posts/machine_learning/nlp/rnn.png)

上图中左边是RNN模型没有按时间展开的图，如果按时间序列展开，则是上图中的右边部分。我们重点观察右边部分的图。

这幅图描述了在序列索引号t附近RNN的模型。其中

* x^{(t)}表示在t时刻训练样本的输入，$ x^{(t-1)}和$ x^{(t+1)}表示在t-1时刻和t+1时刻训练样本的输入。
* h^{(t)}表示在t时刻模型的隐藏状态，它由x^{(t)}和h^{(t-1)}共同决定。
* o^{(t)}表示在t时刻模型的输出，它仅仅由模型当前的隐藏状态h^{(t)}来决定。
* y^{(t)}表示在t时刻模型的真实输出。
* L^{(t)}表示在t时刻模型的损失函数。
* U,W和V是模型的线性关系参数，在整个RNN网络中是共享的。这个是和DNN的不同之处，正是由于共享，体现了RNN模型的“循环反馈”的思想。

### RNN的前向传播算法

任一时刻t，隐藏状态h^{(t)}由x^{(t)}和h^{(t-1)}共同决定。

h^{(t)} = \sigma(z^{(t)}) = \sigma(Ux^{(t)} + Wh^{(t-1)} +b )

其中\sigma为RNN的激活函数，一般为tanh, b为线性关系的偏倚。

任一时刻t，模型的输出o^{(t)}仅仅由h^{(t)}来决定。

o^{(t)} = Vh^{(t)} +c

任一时刻t，模型的预测输出为：

\hat{y}^{(t)} = \sigma(o^{(t)})

通常由于RNN是识别类的分类模型，所以上面这个激活函数一般是softmax。

### RNN的反向传播算法

RNN反向传播算法的思路和DNN是一样的，即通过梯度下降法一轮轮的迭代，得到合适的RNN模型参数U,W,V,b,c。

但由于我们是基于时间反向传播，所以RNN的反向传播有时也叫做BPTT(back-propagation through time)。

重点是这里的BPTT和DNN也有很大的不同点，即**这里所有的U,W,V,b,c在序列的各个位置是共享的，反向传播时我们更新的是相同的参数**。

## LSTM

### 为什么要有LSTM?

RNN虽然理论上可以很漂亮的解决序列数据的训练，但是它也像DNN一样有梯度消失的问题，当序列很长的时候问题尤其严重。

RNN虽然被设计成可以处理整个时间序列信息，但其记忆最深的还是最后输入的一些信号，而更早之前的信号强度越来越低，最后仅仅起到辅助作用。

### LSTM的网络结构

RNN和LSTM的网络结构的对比：

![](/images/posts/machine_learning/nlp/rnn1.png)

![](/images/posts/machine_learning/nlp/lstm.png)

由以上两幅图可以看出来，RNN只有一个神经元和一个tanh层进行重复的学习。但是在长环境中相关的信息和预测的词之间的间隔可以是非常长的，理论上RNN也可以学习到这些知识，但是实践中RNN并不能学到这些知识。

标准LSTM模型是一种特殊的RNN类型，在每一个重复的模块中有四个特殊的结构，以一种特殊的方式进行交互。黄色的矩阵即为学习到的神经网络层。**LSTM模型的核心思想是“细胞状态”**。

遗忘门：
第一步是决定我们从“细胞”中丢弃什么信息，这个操作由一个忘记门层来完成。该层读取当前输入x和前神经元信息h，由ft来决定丢弃的信息。输出结果1表示“完全保留”，0 表示“完全舍弃”。

![](/images/posts/machine_learning/nlp/forget.png)

输入门：
第二步是确定细胞状态所存放的新信息，这一步由两层组成。sigmoid层作为“输入门层”，**决定我们将要更新的值i**；tanh层来**创建一个新的候选值向量**~Ct加入到状态中。在语言模型的例子中，我们希望增加新的主语到细胞状态中，来替代旧的需要忘记的主语。 

![](/images/posts/machine_learning/nlp/input.png)

第三步就是更新旧细胞的状态，将Ct-1更新为Ct。我们把旧状态与ft相乘，丢弃掉我们确定需要丢弃的信息。接着加上it * ~Ct，这就是新的候选值。

![](/images/posts/machine_learning/nlp/output.png)

输出门：

最后一步就是确定输出了，这个输出将会基于我们的细胞状态，但是也是一个过滤后的版本。我们**运行一个sigmoid层来确定细胞状态的哪个部分将输出出去**。

![](/images/posts/machine_learning/nlp/output2.png)

## GRU

### 为什么要有GRU?

GRU即Gated Recurrent Unit。GRU保持了LSTM的效果同时又使结构更加简单，所以它也非常流行。

### GRU的网络结构

![](/images/posts/machine_learning/nlp/gru2.png)

而GRU模型如上图所示，相比LSTM有三个门它只有两个门了，分别为更新门和重置门，即图中的zt和rt。更新门用于控制前一时刻的状态信息被带入到当前状态中的程度，更新门的值越大说明前一时刻的状态信息带入越多。重置门用于控制忽略前一时刻的状态信息的程度，重置门的值越小说明忽略得越多。

r\_t=\sigma(W\_r\cdot[h\_{t-1},x\_t])

z\_t=\sigma(W\_z\cdot[h\_{t-1},x\_t])

{\tilde{h}}\_t=\tanh(W\_{\tilde{h}} \cdot[r\_t\ast h\_{t-1},x\_t])

h\_t=(1-z\_t)\ast h\_{t-1}+z\_t \ast \tilde{h}\_t

y\_t = \sigma(W\_o \cdot h\_t)

## 参考

[RNN](https://www.cnblogs.com/pinard/p/6509630.html)

[RNN梯度下降](http://www.cnblogs.com/xweiblogs/p/5914622.html#undefined)

[LSTM和GRU](https://blog.csdn.net/lreaderl/article/details/78022724)

[GRU](https://blog.csdn.net/wangyangzhizhou/article/details/77332582)

[LSTM](https://zybuluo.com/hanbingtao/note/581764)

[LSTM超参数训练trick](https://www.jianshu.com/p/dcec3f07d3b5)
