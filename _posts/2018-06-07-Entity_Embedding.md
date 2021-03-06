---
layout: post
title: 实体嵌入(Entity Embedding)
categories: [NLP]
description: some word here
keywords: NLP, 实体嵌入
---

## 背景

这篇文章关注的是深度学习领域一个并不非常广为人知的应用领域：结构化数据。

在机器学习/深度学习或任何类型的预测建模任务中，都是先有数据然后再做算法/方法。这也是某些机器学习方法在解决某些特定任务之前需要做大量特征工程的主要原因，这些特定任务包括图像分类、NLP 和许多其它**非常规的数据的处理**——这些数据不能直接送入logistic回归模型或随机森林模型进行处理。相反，深度学习无需任何繁杂和耗时的特征工程也能在这些类型的任务取得良好的表现。大多数时候，这些特征需要领域知识、创造力和大量的试错。

由于特征生成(比如CNN的卷积层)的本质和能力很复杂，所以深度学习在各种各样的图像、文本和音频数据问题上得到了广泛的应用。这些问题无疑对人工智能的发展非常重要，而且这一领域的顶级研究者每年都在分类猫、狗和船等任务上你追我赶，每年的成绩也都优于前一年。但在实际行业应用方面我们却很少看到这种情况。这是为什么呢？**公司企业的数据库涉及到结构化数据，这些才是塑造了我们的日常生活的领域**。

首先，让我们先定义一下结构化数据。**在结构化数据中，你可以将行看作是收集到的数据点或观察，将列看作是表示每个观察的单个属性的字段**。比如说，来自在线零售商店的数据有表示客户交易事件的列和包含所买商品、数量、价格、时间戳等信息的列。

接下来我们谈谈如何将神经网络用于结构化数据任务。实际上，在理论层面上，创建带有任何所需架构的全连接网络都很简单，然后使用列作为输入即可。在损失函数经历过一些点积和反向传播之后，我们将得到一个训练好的网络，然后就可以进行预测了。

尽管看起来非常简单直接，但在处理结构化数据时，人们往往更偏爱基于树的方法，而不是神经网络。原因为何？这可以从算法的角度理解——算法究竟是如何对待和处理我们的数据的。**人们对结构化数据和非结构化数据的处理方式是不同的**。非结构化数据虽然是非常规的，但我们通常处理的是单位量的单个实体，比如像素、体素、音频频率、雷达反向散射、传感器测量结果等等。而对于结构化数据，我们往往需要处理多种不同的数据类型；**这些数据类型分为两大类：数值数据和类别数据**。类别数据需要在训练之前进行预处理，因为包含神经网络在内的大多数算法都还不能直接处理它们。

**编码变量有很多可选的方法，比如标签/数值编码和one-hot编码**。但在内存方面和类别层次的真实表示方面，这些技术还存在问题。内存方面的问题可能更为显著，我们通过一个例子来说明一下。**假设我们列中的信息是一个星期中的某一天。如果我们使用one-hot或任意标签编码这个变量，那么我们就要假设各个层次之间都分别有相等和任意的距离/差别。但这两种方法都假设每两天之间的差别是相等的，但我们很明显知道实际上并不是这样，我们的算法也应该知道这一点**!

神经网络的连续性本质限制了它们在类别变量上的应用。因此，用整型数表示类别变量然后就直接应用神经网络，不能得到好的结果。

**基于树的算法不需要假设类别变量是连续的，因为它们可以按需要进行分支来找到各个状态**，但神经网络不是这样的。**实体嵌入(entity embedding)可以帮助解决这个问题**。实体嵌入可用于**将离散值映射到多维空间中**，其中具有相似函数输出的值彼此靠得更近。比如说，如果你要为一个销售问题将各个省份嵌入到国家这个空间中，那么相似省份的销售就会在这个投射的空间相距更近。

## 实体嵌入(Entity Embedding)

尽管人们对实体嵌入有不同的说法，但它们与我们在词嵌入上看到的用例并没有太大的差异。毕竟，我们只关心我们的分组数据有更高维度的向量表示；这些数据可能是词、每星期的天数、国家等等。**这是从词嵌入到元数据嵌入(在我们情况中是类别)的转换**。

为了**处理由客户ID、出租车ID、日期和时间信息组成的离散的元数据**，我们使用该模型为这些信息中的每种信息联合学习了嵌入。其中每个词都映射到了一个固定大小的向量空间(这种向量被称为词嵌入)。

注意：**根据经验，应该保留没有非常高的基数的类别**。因为如果一个变量的某个特定层次占到了90%的观察，那么它就是一个没有很好的预测价值的变量，我们可能最好还是避开它。

有几点值得关注：

* 店铺所在地的嵌入向量在用TSNE投影到两维空间后和地图位置有着极大的相似性。
* **使用嵌入后的向量可以提高其他算法（KNN、随机森林、gdbt）的准确性**。
* 作者探索了embedding和度量空间之间的联系，试图从数学层面深入探讨embedding的作用。

实体嵌入的优点：

* 实体嵌入解决了独热编码的缺点。具有许多类别的独热编码变量会导致非常稀疏的向量，这在计算上是无效的，并且难以优化。标签编码解决了这一问题，但只能用于基于树的模型。
* 嵌入提供有关不同类别之间距离的信息。使用嵌入的优点在于，在神经网络的训练期间，也要训练分配给每个类别的向量。因此，**在训练过程结束时，我们最终会得到一个代表每个类别的向量**。这些训练过的嵌入被可视化，为每个类别提供可视化。在Rossmann销售预测任务中，即使没有为模型提供地理位信息，德国的可视化嵌入显示了与各州地理位置相似的集群。
* **训练好的嵌入可以保存并用于非深度学习模型**。例如，每月训练分类特征的嵌入并保存。通过加载训练好的分类特征嵌入，我们可以使用这些嵌入来训练随机森林或梯度提升树GBT模型。

选择嵌入尺寸：

嵌入尺寸是指代表每个类别的向量长度，并且可以为每个分类特征设置。 类似于神经网络中超参数的微调（tuning），嵌入尺寸的选择没有硬性的规定。在出租车距离预测任务中，每个特征的嵌入尺寸为10。这些特征具有不同的维度，从7（一周的天数）到57106（客户端ID）。为每个类别选择相同的嵌入尺寸是一种简单易行的方法，但可能不是最优的方法。

对于Rossmann商店销售预测任务，研究人员选择1和M之间的一个值（类别数量）-1，最大嵌入尺寸为10。例如，一周中的某天（7个值）的嵌入尺寸为6， 而商店ID（1115个值）的嵌入尺寸为10。但是，作者没有明确的选择1和M-1之间选择的规则。

Jeremy Howard重建了Rossmann竞赛的解决方案，并提出了以下解决方案来选择嵌入尺寸：

```
c is the amount of the categories per future.
embedding_size = (c+1)//2
if embedding_size > 50:
	embedding_size = 50
```

## 参考

[实体嵌入(向量化)：用深度学习处理结构化数据](https://yq.aliyun.com/articles/497115)

[类别特征处理与实体嵌入](https://blog.csdn.net/h4565445654/article/details/78998444)

[Artificial Neural Networks Applied to Taxi Destination Prediction（阅读笔记）20171207](https://blog.csdn.net/m0_38058163/article/details/78737983)
