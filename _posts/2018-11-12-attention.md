---
layout: post
title: Attention机制
categories: [NLP]
description: some word here
keywords: NLP
---

## Soft Attention(参考1 参考3)

由神经机器翻译提出，基本思想是目标语言端的词往往只与源语言端的部分词有关。

## Hard Attention(参考1 参考3)

由神经机器翻译提出，试图找到与某个输出直接对应的源语言的单词，而认为其他源语言句子中其余单词的对其概率为0。它的对齐要求太高，可能会导致负面影响。

一种折中的方式是**先以hard的思想在源语言端找到与目标端词最有可能对其的位置，然后以这个词为中心取窗口，再以soft的思想在这个窗口上求一个对其概率分布**。

**参考3用强化学习方法进行训练**。

## Global Attention && Local Attention(参考1 参考3)

Global Attention：在计算源语言端上下文向量时，考虑Encoder的所有隐层状态。计算和每个隐层状态对应的权重。

Local Attention：对于长句子，Global Attention由于要计算与每个源语言端词的概率，代价比较大。于是有了一种折中方式，Local Attention只计算每个目标端词与部分源语言端词的对其概率。

## Attention和Multi-Head Attentioon(参考4)

Google的一般化Attention思路也是一个编码序列的方案，因此我们也可以认为它跟RNN、CNN一样，都是一个序列编码的层。

Attention的定义：

Attention(Q, K, V) = softmax(QK)V 用Q和多个K进行计算，得到和每个K的归一化的概率，然后用概率乘V。将Q序列进行编码的过程。**如果Q = K = V，那么就是self attention**。

Multi-Head Attention的定义：

这个是Google提出的新概念，是Attention机制的完善。不过从形式上看，它其实就再简单不过了，就是把Q,K,V通过参数矩阵映射一下，然后再做Attention，把这个过程重复做h次，结果拼接起来就行了，可谓“大道至简”了。**所谓“多头”（Multi-Head），就是只多做几次同样的事情（参数不共享），然后把结果拼接**。

[Attention的tensorflow实现(里面还有position embedding的实现)](https://github.com/bojone/attention/blob/master/attention_tf.py)

## Multi-step Attention(参考5)

在每一个卷积层都会进行 attention 的操作，得到的结果输入到下一层卷积层，这就是多跳注意机制multi-hop attention。这样做的好处是使得模型在得到下一个主意时，能够考虑到之前的已经注意过的词。

## 参考

[1 神经网络机器翻译Neural Machine Translation(2): Attention Mechanism](https://blog.csdn.net/u011414416/article/details/51057789)

[2 深度学习中的注意力机制](https://blog.csdn.net/TG229dvt5I93mxaQ5A6U/article/details/78422216)

[3 不得不了解的五种Attention模型方法及其应用](https://www.sohu.com/a/242214491_164987)

[4 《Attention is All You Need》浅读（简介+代码）(文章开源代码不错)](https://kexue.fm/archives/4765#Multi-Head%20Attention)

[5 《Convolutional Sequence to Sequence Learning》阅读笔记](https://zhuanlan.zhihu.com/p/26918935)

[从《Convolutional Sequence to Sequence Learning》到《Attention Is All You Need》](https://zhuanlan.zhihu.com/p/27464080)

[基于CNN的Seq2Seq模型-Convolutional Sequence to Sequence](https://www.jianshu.com/p/ab949c66271e)
