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

在每一个卷积层都会进行 attention 的操作，得到的结果输入到下一层卷积层，这就是多跳注意机制multi-hop attention。这样做的好处是**使得模型在得到下一个注意时，能够考虑到之前的已经注意过的词**。

[tf实现](https://github.com/tobyyouup/conv_seq2seq)

[代码存疑，可能有问题 pytorch实现](https://github.com/pengshuang/CNN_Seq2Seq)

[从《Convolutional Sequence to Sequence Learning》到《Attention Is All You Need》](https://zhuanlan.zhihu.com/p/27464080)

[Convolutional Sequence to Sequence Learning 阅读和实现](https://zhuanlan.zhihu.com/p/51952607)

[《Convolutional Sequence to Sequence Learning》阅读笔记](https://zhuanlan.zhihu.com/p/26918935)

### Convolutional Block Structure

encoder和decoder都是由l层卷积层构成。“卷积计算+非线性计算”看作一个单元Convolutional Block，这个单元在不同block structure中是共享的（也就是说l层卷积共享参数）。卷积计算和非线性计算，会有一个残差连接。

非线性计算部分：输入拆分成两个大小相同的矩阵，然后一个矩阵不变，另一个矩阵作为门控，控制着网络中的信息流，即哪些能够传递到下一个神经元中。

### Multi-step Attention

解码器的每一层都使用了attention机制，所以叫multi-step attention。

attention的权重由decoder的当前输出hi和encoder的最后一层输出来决定，利用得到的权重对encoder的最后一层输出进行加权，得到表示输入句子信息的向量ci。

解码器看源码感觉是：**每个层的卷积层参数共享；输入通过卷积层得到输出，再经过非线性计算得到hi，然后经过attention得到ci，再把ci和hi加起来，作为下一层输入**。

## attention用于生成图片的描述(参考6)

attention有很多是在机器翻译领域的应用，但attention机制同样也能应用于递归模型。在Show，Attend and Tell一文中，作者**将attention机制应用于生成图片的描述**。他们用卷积神经网络来“编码”图片，并用一个递归神经网络模型和attention机制来生成描述。通过对attention权重值的可视化（就如之前机器翻译的例子一样），在生成词语的同时我们能解释模型正在关注哪个部分。

## 参考

[1 神经网络机器翻译Neural Machine Translation(2): Attention Mechanism](https://blog.csdn.net/u011414416/article/details/51057789)

[2 深度学习中的注意力机制](https://blog.csdn.net/TG229dvt5I93mxaQ5A6U/article/details/78422216)

[3 不得不了解的五种Attention模型方法及其应用](https://www.sohu.com/a/242214491_164987)

[4 《Attention is All You Need》浅读（简介+代码）(文章开源代码不错)](https://kexue.fm/archives/4765#Multi-Head%20Attention)

[5 《Convolutional Sequence to Sequence Learning》阅读笔记](https://zhuanlan.zhihu.com/p/26918935)

[6 深度学习方法（九）：自然语言处理中的Attention Model注意力模型](https://blog.csdn.net/xbinworld/article/details/54607525)

[从《Convolutional Sequence to Sequence Learning》到《Attention Is All You Need》](https://zhuanlan.zhihu.com/p/27464080)

[基于CNN的Seq2Seq模型-Convolutional Sequence to Sequence](https://www.jianshu.com/p/ab949c66271e)
