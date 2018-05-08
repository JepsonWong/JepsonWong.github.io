---
layout: post
title: 利用CNN、RNN来进行文本分类，探索句子建模
categories: [NLP, 深度学习]
description: 介绍CNN、RNN在NLP中的一些应用
keywords: NLP, 深度学习, 文本分类
---

## 1. 字符级模型

### 1.1 16年文章Neural Machine Translation in Linear Time

对卷积层的改进，提出了扩张卷积神经网络(Dilated Convolution)应用与机器翻译领域。Dilated convolution实际上**要解决的问题是池化层的池化会损失很多信息（无论该信息是有用还是无用）**。Dilated convolution的主要贡献就是，**如何在去掉池化操作的同时，而不降低网络的感受野**。

在该模型中，**句子建模时输入是以句子的字符级别开始的**，之后随着卷积核所能覆盖的范围扩展，不断地去交互信息，同时还能够保证原始的输入信息不被丢失。

[《Neural Machine Translation in Linear Time》阅读笔记](https://zhuanlan.zhihu.com/p/23795111)

### 1.2 16年文章Character-level Convolutional Networks for Text Classification

字符级CNN。没有做变长的convolutional处理。

[字符级卷积神经网络（Char-CNN）实现文本分类--模型介绍与TensorFlow实现 6个卷积层+3个全连接层](http://www.h3399.cn/201704/78031.html)

## 2. 单词级模型

### 2.1 14年文章Convolutional Neural Networks for Sentence Classification

**其中的Pooling层解决了可变长度的句子输入问题**。

The input layer is a sentence comprised of concatenated word2vec word embeddings. That’s followed by a convolutional layer with multiple filters, then a max-pooling layer, and finally a softmax classifier.

The paper also experiments with two different channels in the form of static and dynamic word embeddings, where one channel is adjusted during training and the other isn’t.

卷积的应用：位置不变性和组合性

在图像中卷积核通常是对图像的一小块区域进行计算，而在文本中，一句话所构成的词向量作为输入。每一行代表一个词的词向量，所以在处理文本时，**卷积核通常覆盖上下几行的词，所以此时卷积核的宽度与输入的宽度相同**，通过这样的方式，我们就能够**捕捉到多个连续词之间的特征**，并且能够在同一类特征计算时中共享权重。

[论文《Convolutional Neural Networks for Sentence Classification》总结](https://blog.csdn.net/rxt2012kc/article/details/73739756)

### 2.2 16年文章A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification

对文章中的模型Convolutional Neural Networks for Sentence Classification进行各种各样的对比实验，得到一些超参数的设置经验。

* 对于预训练的词向量（glove， word2vec）而言，二者对不同分类任务各有优劣，但效果都比one-hot编码要强（虽然one-hot编码方式在文本分类任务中取得了比较好的效果）。
* 卷积核的窗口大小对实验结果有着比较重要的影响。首先，ws在1-10之间取值较好，且如果训练集中的句子长度较大（100+）时，我们可以考虑使用较大的ws以获取上下文关系。其次，对不同尺寸ws的窗口进行结合会对结果产生影响。当把与最优ws相近的ws结合时会提升效果，但是如果将距离最优ws较远的ws相结合时会损害分类性能。一般取为3-5。
* 卷积核数量num_filters也对实验结果比较重要。最好不要超过600，超过600可能会导致过拟合。一般设为100-200。
* pooling方式就使用1-max就可以。mean或者k-max pooling效果都不太好。
* l2正则化效益很小，相比而言，dropout在神经网络中有着广泛的使用和很好的效果，dropout一般设为0.5
* 激活函数的话，目前广泛应用的是ReLU、tanh函数。

[textcnn源码1](https://github.com/dennybritz/cnn-text-classification-tf)

[textcnn源码2](https://github.com/yoonkim/CNN_sentence)

[Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

[CNN用于句子分类时的超参数分析](https://blog.csdn.net/liuchonge/article/details/67040089)

[Text-CNN 文本分类: 超参数选择](https://blog.csdn.net/chuchus/article/details/77847476)

[论文Convolutional Naural Networks for Sentence Classification--模型介绍篇](https://blog.csdn.net/liuchonge/article/details/60328365)

[论文Convolutional Naural Networks for Sentence Classification--TensorFlow实现篇](https://blog.csdn.net/liuchonge/article/details/60333323)

### 2.3 14年文章A Convolutional Neural Network for Modelling Sentences

**DCNN能够处理可变长度的输入**。

池化的应用：降维和保留显著的特征

很多时候我们会选择最大池化，但本论文对句子级别特征的池化过程进行了改进并且提出了DCNN动态卷积网络（Dynamic Convolutional Neural Network）。

Max-pooling最为常见，最大池化是取整个区域的最大值作为特征，在自然语言处理中**常用于分类问题**，希望观察到的特征是强特征，以便可以区分出是哪一个类别。Average-pooling**通常是用于主题模型**，常常是一个句子不止一个主题标签，如果是使用Max-pooling的话信息过少，所以使用Average的话可以广泛反映这个区域的特征。最后两个K-max pooling是选取一个区域的前k个大的特征。Dynamic pooling是根据网络结构动态调整取特征的个数。**最后两个的组合选取，就是该篇论文的亮点**。

该论文的亮点首先对句子语义建模，在底层通过**组合邻近的词语信息**，逐步向上传递，**上层则又组合新的语义信息**，从而**使得句子中相离较远的词语也有交互行为（或者某种语义联系）**。从直观上来看，这个模型能够通过词语的组合，再通过池化层提取出句子中重要的语义信息。

另一个亮点就是在池化过程中，该模型采用动态k-Max池化，这里池化的结果不是返回一个最大值，而是返回k组最大值，这些最大值是原输入的一个子序列。池化中的参数k可以是一个动态函数，具体的值依赖于输入或者网络的其他参数。

模型还有一个Folding层（折叠操作层）。这里考虑相邻的两行之间的某种联系，将两行的词向量相加。

该模型亮点很多，总结如下，首先它保留了句子中词序和词语之间的相对位置，同时考虑了句子中相隔较远的词语之间的语义信息，通过动态k-max pooling较好地保留句子中多个重要信息且根据句子长度动态变化特征抽取的个数。

[《A Convolutional Neural Network for Modelling Sentences 》閱讀筆記](https://www.getit01.com/p2018020129925124/)

[《A Convolutional Neural Network for Modelling Sentences 》阅读笔记](https://zhuanlan.zhihu.com/p/29925124)

[CNN与句子分类之动态池化方法DCNN--模型介绍篇](https://blog.csdn.net/liuchonge/article/details/67638232)

[A convolutional Neural Network for Modelling Sentences](https://blog.csdn.net/alwaystry/article/details/53840736)

### 2.4 16年北京工业大学专利一种基于卷积神经网络与随机森林的短文本分类方法

[一种基于卷积神经网络与随机森林的短文本分类方法](https://patentimages.storage.googleapis.com/5b/5c/83/3222cd69226244/CN107066553A.pdf)

[一种基于卷积神经网络与随机森林的短文本分类方法](https://patents.google.com/patent/CN107066553A/zh)

输入层 + 卷积层 + 池化层 + 卷积层 + 随机森林层

## 3. 其他句子建模的创新

### 3.1 15年文章Dependency-based Convolutional Neural Networks for Sentence Embedding

### 3.2 14年文章Convolutional Neural Network Architectures for Matching Natural Language Sentences

## github

[Text Classification with CNN and RNN](https://github.com/gaussic/text-classification-cnn-rnn)

## 参考

[从CNN视角看在自然语言处理上的应用](http://www.chuangyejia.vip/article/detail/209441.html) 

[Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
