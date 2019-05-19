---
layout: post
title: Text Summarization
categories: [NLP]
description: some word here
keywords: NLP, Text Summarization
---

## 文本摘要

### 基于抽取式的方法（Extractive）

基于Page Rank，把每句话当作一个节点（Page Rank中功德节点式某个网页），找到Page Rank值大的几句话。

句子向量的生成有多种方法。

图的生成也有多种方法，权重可以采用句子间的相似度，也可以删去某些权重比较低的连接。

提取得分高的前几个句子，也不一定最好，因为可能这几个句子相似，我们要句子有代表性又多样化。

### 基于生成式的方法（Abstractive）

基于sequence模型，然后加一些attention model等。最常见的是seq2seq模型。

# 博客

[一文梳理NLP之机器翻译和自动摘要的发展现状](https://cloud.tencent.com/developer/article/1165357)

[文本自动摘要](https://www.jianshu.com/p/0cf99a70b059)

[基于TextRank的中文摘要工具](http://jayveehe.github.io/2016/05/11/da_textrank/)

[netwotkx 摘要](https://www.google.com/search?ei=2_6vXLqpAcy4mAXW8KHIDQ&q=netwotkx+%E6%91%98%E8%A6%81&oq=netwotkx+%E6%91%98%E8%A6%81&gs_l=psy-ab.12...0.0..3151...0.0..0.0.0.......0......gws-wiz.4w2LSO_k36s)

# 源码

[使用Python实现一个文本自动摘要工具](https://zhuanlan.zhihu.com/p/30603995)

[文章自动摘要之Pointer-Generator Networks（没看）](https://blog.csdn.net/qq_17827079/article/details/80171728)

[keyphrase github](https://github.com/starevelyn/gluon-nlp/tree/master/scripts/keyphrase)

[Generating wikipedia by summarizing long sequences（没看）](https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=1&rsv_idx=2&tn=baiduhome_pg&wd=Generating%20wikipedia%20by%20summarizing%20long%20sequences&oq=Generating%2520wikipedia%2520by%2520summarizing%2520long%2520sequences&rsv_pq=e738c1890003a81d&rsv_t=85c1jim04xFnUbfmKUmIf4QzotD7uPuvYY0juFXDsKz%2FkaaGQ%2F8VBE%2FH28PgUXLk3dhg&rqlang=cn&rsv_enter=0)

[变身抓重点小能手：机器学习中的文本摘要入门指南](https://zhuanlan.zhihu.com/p/63402103)

[谷歌开源新的TensorFlow文本自动摘要代码](https://www.jiqizhixin.com/articles/2016-08-25-4)

# 论文

[Pointer Networks简介及其应用](https://zhuanlan.zhihu.com/p/48959800)

https://github.com/yuedongP
博主https://www.cs.mcgill.ca/~ydong26/
Yue Dong. "A Survey on Neural Network-Based Summarization Methods." arXiv preprint arXiv:2199934 (2018)
Yue Dong, Yikang Shen, Eric Crawford, Herke van Hoof, Jackie Chi Kit Cheung. "BanditSum: Extractive Summarization as a Contextual Bandit." EMNLP (2018)

A Survey on Neural Network-Based Summarization Methods Yue Dong
Learning Multi-task Communication with Message Passing for Sequence Learnin Yue dong
