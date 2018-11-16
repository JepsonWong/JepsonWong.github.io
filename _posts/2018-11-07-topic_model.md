---
layout: post
title: 主题模型
categories: [机器学习]
description: some word here
keywords: 主题模型
---

## LDA: latent dirichlet allocation

考虑两个句子的语义。而不是生硬的以单词出现与否等来作为句子的特征，因为这样没有考虑到一词多义和一义多次的问题。

它是一个生成模型，他认为一篇文章的每个词都是通过“以一定概率选择了某个主题，并从这个主题中以一定概率选择某个词语”这样一个过程得到的。

### 如何生成M份包含N个单词的文档

#### unigram model

通过训练预料获得一个单词的概率分布函数，然后根据这个概率分布函数每个生成一个单词，通过这种方法M次生成M个文档。（单词的概率分布函数可以通过预料进行统计学习得到）

#### Mixture of unigram

unigram的缺点是生成的文本没有主题，比较简单。根据主题的概率分布产生主题，然后由主题对应的单词概率分布生成单词，根据这种方法来生成文档。但这种方法主题只选择一次。

#### LDA

按照先验概率p(di)选择一篇文档；从狄利克雷分布（即Dirichlet分布）中取样生成**文档di的主题分布**，换言之，主题分布由超参数为theta的Dirichlet分布生成；从主题的多项式分布中取样生成文档di第j个词的主题zij；从狄利克雷分布（即Dirichlet分布）中取样生成主题对应的词语分布，换言之，词语分布由参数为beta的Dirichlet分布生成；从词语的多项式分布中采样最终生成词语wij。

## pLSA: probabilistic latent semantic analysis

频率派思想，参数未知但固定。

LDA在pLSA的基础上为主题分布和词分布分别加了两个Dirichlet先验。即pLSA中当确定了一篇文档主题分布和词分布是唯一确定的；而LDA主题分布和词分布不再唯一确定不变，但再怎么变化，也依然服从一定的分布，即主题分布跟词分布**由Dirichlet先验随机确定**。

## Gaussian LDA

没看，用到再说吧。

[Gaussian LDA(2): Gaussian LDA简介](https://blog.csdn.net/u011414416/article/details/51188483)

## 参考

[深入理解LDA和pLSA](https://blog.csdn.net/u010159842/article/details/48637095)

[话题学习-LDA学习](https://yq.aliyun.com/wenji/234616?spm=a2c4e.11155472.blogcont.35.49af4370cCXNJP)

[主题模型︱几款新主题模型——SentenceLDA、CopulaLDA、TWE简析与实现](https://yq.aliyun.com/wenji/265159)

[Gaussian LDA(1): LDA回顾以及变分EM](https://blog.csdn.net/u011414416/article/details/51168242)

[通俗理解LDA主题模型](https://blog.csdn.net/v_july_v/article/details/41209515)

[LDA PLSA 比较总结](https://blog.csdn.net/menyangyang/article/details/44080303)
