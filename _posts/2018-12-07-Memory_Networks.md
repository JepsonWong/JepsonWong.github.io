---
layout: post
title: Memory Networks系列
categories: [NLP]
description: some word here
keywords: NLP, memory network
---

## Memory Networks(2015)

[Memory Networks原理及其代码解析 文章参数d为3倍的lenW](https://blog.csdn.net/u011274209/article/details/53384232?ref=myread)

[读论文写论文——MEMORY NETWORKS 参数d为3倍的lenW](https://blog.csdn.net/qfnu_cjt_wl/article/details/53909079)

[讲的不错，也有代码 Gated End-to-End Memory Networks的简介和python实现](https://zhuanlan.zhihu.com/p/53075774)

传统的RNN等模型使用隐藏状态、连接参数、Attention机制等实现记忆功能，但有三方面劣势：记忆量非常非常有限，在对话系统、阅读理解等需要较多上下文的任务中并不适用；记忆时间有限，一般可能也就十几到几十个步长，并不适合推理类任务、长时间依赖任务；把文本转换成vector时丢失了大量信息，导致记忆并不精准，同样不适用于推理类等任务。

所以，就出现了Memory NN和NTM（神经图灵机）。

Memory Networks是一类**模型框架**，组件I、G、R、O可以使用不同的实现。

**文章提出了一个通用的解决长期记忆问题的算法框架, 框架中的每一个模块都可以变更成新的实现, 可以根据不同的应用场景进行适配**。

memory network包含一个存储器m（由mi索引的对象数组，可以是vector数组或者string数组）和四个（潜在学习）组件I、G、O和R。

* I：（input feature map）将输入转换成内部特征表达。**文中使用向量空间模型，每个维度以该词的词频为大小**。
* G：（generalization）在给定新输入的情况下更新旧记忆。我们成它为generalization的原因是网络可以在这个阶段压缩或者概括它的记忆，以用于某些预期的未来用途。文章中**对G组件进行了简化**，仅仅对输入的特征表示进行了存储，对于更新等复杂任务没有太多涉及，将其作为了G的潜力。基础模型中，直接将输入存入到M中。
* O：（output feature map）在给定新输入和当前的存储状态的情况下，生成新输出。
* R：（response）将输出转换为所需的响应格式。例如，文本响应或者动作。

训练和测试阶段的不同在于测试阶段I、G、O、R的参数不会再更新，但是存储器m依然会存储信息。

[Answering Reading Comprehension Using Memory Networks](http://cs224d.stanford.edu/reports/KapashiDarshan.pdf)

本文的一些要点：

* 本文虽然提出了一种泛化的的模型架构（Memory Network），但是很多部分并没有做得很完善，论文接下来也**对输入词、memory比较大、新词等问题进行了专门的阐述**。
* 本模型**不是端到端训练的过程**（因为要求**训练集必须有支持事实这一事实**），所以可以应用的场景较少，因此接下来Facebook团队就提出了端到端的记忆网络，这在End-To-End Memory Networks中会详细说明并且实现。**这里有两个监督的部分，一个是支撑记忆，一个是正确答案，是一个监督算法；以后改成了弱监督，以N2N为代表**。
* 其中O和R部分使用了相同的**函数S(x, y)，但是参数不一样**。**参考1中有这个公式**。O和S函数有参数U，U是一个权重矩阵（权重矩阵参数不一样），维度为n * D。D是输入向量的维度，n是嵌入的维度，**n是一个超参数**。作者使用D = 3 * len(W)的维度。这里考虑了3倍向量空间模型长度，三份直接连接在一起就可以了。

## End-To-End Memory Networks(2015)

文章提出了一个可以**端到端训练**的Memory Networks，并且在训练阶段比原始的Memory Networks需要更少的监督信息（**减少生成答案时需要事实依据的监督项**），在实际应用中意义重大。

[记忆网络之End-To-End Memory Networks 文章分析的很好](https://blog.csdn.net/liuchonge/article/details/78093567)

[记忆网络之End-To-End Memory Networks 没看](https://zhuanlan.zhihu.com/p/29679742)

### 模型讲解

模型的结构有两种，一种是单层的模型，一种是多层模型。

#### 单层记忆网络

输入：两部分，一部分为输入集合S = {x1, x2, ……, xn}，一部分为输入向量Q（表示问题）。

输出：一部分，使用输出向量a表示预测向量。

参数：A、B、C、W四个矩阵，A、B、C是Embedding矩阵，将输入文本和Q编码成词向量，W书最终的输出矩阵。

过程：记忆网络模型通过对上下文集合S和问题向量Q进行数学变换，得到对应问题的答案。具体过程是：对于输入的句子S**分别会使用A和C进行编码得到Input和Output的记忆模块**，Input用来跟Q编码得到的向量相乘得到**每句话跟Q的相关性**，Output则与该相关性进行加权求和得到输出向量。然后再加上Q并传入最终的输出层。

输入模块：对应Memory Networks的I、G组件，作用是将输入的文本转化成向量并且存在memory中。BOW、位置、时序编码。

输出模块：分为Input和和Output模块，一个用于跟Q相互作用得到各个memory slot和问题的相关度，一个使用该信息产生输出。输出o。

Response模块：根据输出模块输出的信息产生最终的答案。

#### 多层记忆网络

多层记忆网络和单层基本看结构是类似的，有一些小的细节需要改变，将几个单层网络连接起来。

为了减少参数，可以对多层A、C参数进行共享或者进行其他策略。

### 模型实现

可以针对QA任务的bAbI和语言建模的PTB来实现该模型。

其中QA任务github我已经实现，地址为：https://github.com/JepsonWong/Keras-Examples/tree/master/Sequential_Functional_Model/%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3%E6%9C%BA%E5%99%A8%E4%BA%BA

也有开源的：

[PTB语言模型建模 tf版本](https://github.com/carpedm20/MemN2N-tensorflow)

[bAbI建模 tf版本](https://github.com/domluna/memn2n)

下面重点解释一下PTB语言模型建模的代码说明。

这个任务是一个很传统的语言建模任务，其实就是给定一个词序列预测下一个词出现的概率。

* 由于输入是单词级别，不再是QA任务中的句子，所以不需要句子向量的编码，直接把每个单词的词向量存入memory即可。
* 这里不存在Q，每个训练数据的Q都是一样的，所以直接将Q向量设置为0.1的常量，不需要进行embedding操作。
* 因为之前都是使用LSTM来进行语言建模，所以为了让模型更像RNN，我们采用第二种参数绑定方式，也就是让每层的A和C保持相同，使用H矩阵来对输入进行线性映射。
* 文中提出要对每个hop中一般的神经元进行ReLU非线性处理。
* 采用更深的模型，hop=6或者7，而且memory size也变得更大，100。

```
# 定义模型输入的placeholder，input对应Question，后面会初始化为0.1的常向量，time是时序信息，后面会按照其顺序进行初始化，注意其shape是batch_size * mem_size，因为它对应的memory中每句话的时序信息，target是下一个词，及我们要预测的结果，context是上下文信息，就是要保存到memory中的信息。

self.input = tf.placeholder(tf.float32, [None, self.edim], name="input")
self.time = tf.placeholder(tf.int32, [None, self.mem_size], name="time")
self.target = tf.placeholder(tf.float32, [self.batch_size, self.nwords], name="target")
self.context = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="context")

# 定义变量，A对应论文中的A，B对应论文中的C，C对应论文中的H矩阵，这里作者并未按照论文中变量的命名规则定义
self.A = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
self.B = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
self.C = tf.Variable(tf.random_normal([self.edim, self.edim], stddev=self.init_std))

# Temporal Encoding，时序编码矩阵，T_A对应T_A，T_B对应T_C
self.T_A = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))
self.T_B = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))
```

## Ask Me Anything: Dynamic Memory Networks for Natural Language Processing

## THE GOLDILOCKS PRINCIPLE: READING CHILDREN’S BOOKS WITH EXPLICIT MEMORY REPRESENTATIONS

## Key-Value Memory Networks for Directly Reading Documents(2016)

## TRACKING THE WORLD STATE WITH RECURRENT ENTITY NETWORKS(2017)

## QANet模型

[QANet模型](https://blog.csdn.net/qq_34499130/article/details/80282999)

[QANet Tensoerflow github](https://github.com/NLPLearn/QANet)

[QANet: Combining Local Convolution with Global Self-Attention for Reading Co](https://www.baidu.com/s?wd=QANet%3A%20Combining%20Local%20Convolution%20with%20Global%20Self-Attention%20for%20Reading%20Co&rsv_spt=1&rsv_iqid=0xfc7441a9000286f4&issp=1&f=8&rsv_bp=0&rsv_idx=2&ie=utf-8&tn=baiduhome_pg&rsv_enter=1&rsv_n=2&rsv_sug3=1&rsv_sug2=0&inputT=819&rsv_sug4=819)

## Github开源实现

[MemNN](https://github.com/facebook/MemNN)

## 参考

[1 对话系统专栏](https://blog.csdn.net/Irving_zhang/column/info/18405)

[2 记忆网络Memory Network系列](https://blog.csdn.net/Irving_zhang/article/details/79094416)

[3 记忆网络系列 没看](https://zhuanlan.zhihu.com/c_129532277)

[对话系统（没看）](https://blog.csdn.net/Irving_zhang/column/info/18405)

[经典的端到端聊天模型（没看）](http://www.shuang0420.com/2017/10/05/%E7%BB%8F%E5%85%B8%E7%9A%84%E7%AB%AF%E5%88%B0%E7%AB%AF%E8%81%8A%E5%A4%A9%E6%A8%A1%E5%9E%8B/)

[现在基于深度学习的对话系统常用的评价指标有哪些，分别有什么优缺点和适用范围？（没看）](https://www.zhihu.com/question/264731577)

[深度学习对话系统理论篇--数据集和评价指标介绍（没看）](https://blog.csdn.net/liuchonge/article/details/79104045)

[对话系统系列（没看）](https://blog.csdn.net/liuchonge)

[Memory Networks（没看）](https://blog.csdn.net/amds123/article/details/78797801)

[记忆网络Memory Network（没看）](https://blog.csdn.net/irving_zhang/article/details/79094416)

[记忆网络模型系列之End to End Memory Network（没看）](https://blog.csdn.net/irving_zhang/article/details/79111102)

[记忆网络之Memory Networks（没看）](https://blog.csdn.net/liuchonge/article/details/78082761)

[论文笔记 - Memory Networks 系列（没看）](https://zhuanlan.zhihu.com/p/32257642?edition=yidianzixun&utm_source=yidianzixun&yidian_docid=0HymGR2b)

[Memory Networks原理及其代码解析（没看）](https://blog.csdn.net/u011274209/article/details/53384232?ref=myread)

[记忆网络之Memory Networks（没看）](https://zhuanlan.zhihu.com/p/29590286)

[记忆网络-Memory Network（没看）](https://zhuanlan.zhihu.com/c_129532277)

[End-To-End Memory Networks（没看）](https://blog.csdn.net/u014300008/article/details/52794821)

[A Neural Conversational Model（没看）](http://blog.sina.com.cn/s/blog_15f12409f0102wizw.html)

[Mac Neural Conversational Model 自动聊天机器人实验（没看）](https://ask.julyedu.com/question/7410)

[Sequence to Sequence Learning with Neural Networks（没看）](https://blog.csdn.net/fangqingan_java/article/details/53232030)

