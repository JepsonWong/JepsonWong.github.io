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
* 卷积核数量num\_filters也对实验结果比较重要。最好不要超过600，超过600可能会导致过拟合。一般设为100-200。
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

### 2.5 15年文章Recurrent Convolutional Neural Networks for Text Classification

本文采用周期循环神经网络比卷积神经网络能够更加减少噪声，利用最大池化层选取一句话中最重要的特征。

* 双向循环结构：比传统的基于窗口的神经网络噪声要小，能够最大化地提取上下文信息。
* max-pooling layer池化层：自动决策哪个特征占有更加重要的作用。

模型很简单，介绍如下。

把lstm和cnn结合起来了。举例来说对于词序列: A B C D E F来说，在形成C词的向量化表示的时候，使用的不再仅仅是C的word embedding，而是C左边的内容构成的向量和C以及C右边内容构成的向量的拼接形式。由于使用到了左右两边的内容故使用的是双向的Lstm。使用1-d convolution的方式得到一系列的y，最后经过max-pooling的方式得到整个句子的向量化表示，最后的预测也是基于该句子的。

* 先经过1层双向LSTM，该词的左侧的词正向输入进去得到一个词向量，该词的右侧反向输入进去得到一个词向量。再结合该词的词向量，生成一个\[1, 3 * wordsize\]的词向量(wordsize是单个词向量的长度)，得到y1。
* 再经过全连接层，tanh为非线性函数，得到y2。
* 再经过最大池化层，得出最大化向量y3.
* 再经过全连接层，sigmod为非线性函数，得到最终的多分类。

此前方法存在的缺陷：

* 传统文本文类方法：性能很大程度上依赖于特征的表示和选择，然而，常用的特征的表达方法经常忽略上下文信息、单词在文章中的位置，在获取词汇语义方面达不到令人满意的程度。high-ordern-grams和更为复杂的特征表示方法虽然可以获取上下文信息并且保留词序，但是**存在数据稀疏问题**。
* RecursiveNeuralNetwork递归神经网络：在构建句子表达方面很有效，但是它是基于树结构来表达句意的，性能在很大程度上依赖文本树结构，构建这样的一个树时间复杂度会很高至少是O(n^2)，而且两个句子之间的关系无法通过树的结构表示，所以RecursiveNN不适合对长句子或文档建模。
* RecurrentNeuralNetwork循环神经网络：可以能够很好的捕获上下文信息，但他是**有偏差的模型**，后输入的词要比先输入的词占有更重要的地位，所以在获取整篇文档语义时这种方法就不那么有效了。
* ConvolutionalNeuralNetwork卷积神经网络：是个无偏的模型，对比RNN可以很好的获取文档的语义信息，但是为了简化卷积核都采用固定的窗口，**难点在于确定窗口大小**，太小的窗口可能丢失重要信息，太大的窗口又会扩大参数空间

创新点：

* 提出一个新的模型RecurrentConvolutional NeuralNetwork并将其用到文本分类当中。
* 首先在学习词的表达的时候，采用**双向循环结构获取文本信息**，比传统的基于窗口的神经网络更能减少噪声，而且在学习文本表达时可以大范围的保留词序。
* 其次使用**最大池化层获取文本主要成分**，自动判断哪个特征在文本分类过程中起更重要的作用。

优点：同时利用了RNN和CNN的优点，时间复杂度仅为O(n)，与文本长度呈线性相关。

[论文《Recurrent Convolutional Neural Networks for Text Classification》总结](https://blog.csdn.net/rxt2012kc/article/details/73742362)

[Recurrent Convolutional Neural Networks for Text Classification阅读笔记](https://blog.csdn.net/cat12385/article/details/49104935)

[Recurrent Convolutional Neural Networks for Text Classification(c++源码实现，比较硬气的实现)](https://github.com/pyk/rcnn)

[Keras实现的RCNN文本分类器](https://weibo.com/1402400261/EhBDy16PX?type=comment#_rnd1525851817487)

[RCNN Keras实现版本](https://github.com/airalcorn2/Recurrent-Convolutional-Neural-Network-Text-Classifier/blob/2c4d87d35c99a03a20840acf85ba6583b1340fd4/recurrent_convolutional_keras.py#L39)

[RCNN Keras版本](https://github.com/prernakhurana2/RCNN/blob/master/R-CNN_20NG.py)

[RCNN Tensorflow版本](https://github.com/brightmart/text_classification/blob/master/a04_TextRCNN/p71_TextRCNN_model.py#L51)

[RCNN Tensorflow版本1](https://github.com/zhengwsh/text-classification/blob/master/text_rcnn.py)

### 2.6 TextCNNRNN(CNN+LSTM模型)

[A C-LSTM Neural Network for Text Classification]()

这篇论文里只是用cnn对原文的词向量以**某一长度的filter**进行卷积抽象，这样原来的纯粹词向量序列就变成了经过卷积的抽象含义序列。最后对原句子的encoder还是使用lstm，由于使用了抽象的含义向量，因此其分类效果将优于传统的lstm，这里的cnn可以理解为起到了特征提取的作用。

[multi-class-text-classification-cnn-rnn](https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn)

CNN输入本来是\[batch\_size, seq\_legnth, embedding\_dim, 1\]，padding成\[batch\_size, seq\_legnth + filter\_size - 1, embedding\_dim, 1\]，这样卷积层输出为\[batch\_size, seq\_legnth, 1, num\_filters\]。连接池化层，池化层输出为\[batch\_size, seq\_length/max\_pool\_size的上界, 1, num\_filters\]。然后去掉axis=2的维度，输出shape为\[batch\_size, seq\_length/max\_pool\_size的上界, num\_filters\]。

将三个不同的kernel\_size输出做concat，shape为\[batch\_size, seq\_length/max\_pool\_size的上界, num\_filters\*3\]。作为LSTM的输入。

[Multi-class Text Classification (CNN, LSTM, C-LSTM)](https://github.com/zackhy/TextClassification)

介绍里面的C-LSTM网络结构，CNN输入是\[batch\_size, seq\_legnth, embedding\_dim, 1\]，卷积层输出为\[batch\_size, seq\_legnth - filter\_size + 1, 1, num\_filters\]。注意不同kernel\_size的seq\_legnth - filter\_size + 1不一样，所以后面要弄成一样的，然后去掉axis=2的维度，seq\_legnth - filter\_size + 1变为max\_feature\_length，LSTM的seq\_length为max\_feature\_length。

将三个不同的kernel\_size输出做concat，最后LSTM输入为\[batch\_size, max\_feature\_length, num\_filters\*3\]。

### 2.7 TextRNN

[案例分享|用深度学习(CNN RNN Attention)解决大规模文本分类问题 （二）](https://www.evget.com/article/2017/3/28/26015.html)

尽管TextCNN能够在很多任务里面能有不错的表现，但CNN有个最大问题是固定 filter\_size 的视野，一方面无法建模更长的序列信息，另一方面 filter\_size 的超参调节也很繁琐。CNN本质是做文本的特征表达工作，而自然语言处理中更常用的是递归神经网络(RNN, Recurrent Neural Network)，能够更好的表达上下文信息。具体在文本分类任务中，Bi-directional RNN(实际使用的是双向LSTM)从某种意义上可以理解为可以捕获变长且双向的的 “n-gram” 信息。

## 3. 其他句子建模的创新

### 3.1 15年文章Dependency-based Convolutional Neural Networks for Sentence Embedding

### 3.2 14年文章Convolutional Neural Network Architectures for Matching Natural Language Sentences

## 4. 语音相关的模型

FSMN（feedforward sequential memory networks）

[显著超越流行长短时记忆网络，阿里提出DFSMN语音识别声学模型](https://www.sohu.com/a/225698724_473283)

CFSMN：对FSMN进行改造，模型缩小，速度提高。同时效果比blstm好。

DFSMN：对CFSMN进行改造。

[fsmn](vsooda.github.io/2018/03/12/fsmn/#dfsmn)

## github

[Text Classification with CNN and RNN](https://github.com/gaussic/text-classification-cnn-rnn)

[text-classification](https://github.com/zhengwsh/text-classification)

[text\_classification](https://github.com/brightmart/text_classification)

[Awesome-Text-Classification 未看完](https://github.com/fendouai/Awesome-Text-Classification)

## 参考

[从CNN视角看在自然语言处理上的应用](http://www.chuangyejia.vip/article/detail/209441.html) 

[Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)

[几种使用了CNN（卷积神经网络）的文本分类模型 Learning text representation using recurrent convolutional neural network with highway layers](https://blog.csdn.net/guoyuhaoaaa/article/details/53188918)

[案例分享|用深度学习(CNN RNN Attention)解决大规模文本分类问题 （二）](https://www.evget.com/article/2017/3/28/26015.html)
