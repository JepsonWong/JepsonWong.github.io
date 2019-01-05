---
layout: post
title: 用Google Colab跑TensorFlow Hub模型
categories: [TensorFlow]
description: some word here
keywords: TensorFlow, Google Colab, TensorFlow Hub
---

## 背景

最近突然想跑一个TensorFlow Hub上的预训练模型，但是需要连接到Google来下载模型，而且没有找到从网页预先下载到本地的方法，命令行翻墙比较复杂。所以想到了用Google Colab上免费的GPU和TPU来训练，而且在Google Colab上下载模型也别快。

## Google Colab

[Google Colab基础使用指南](https://blog.csdn.net/sherpahu/article/details/82931761)

[Google Colab——用谷歌免费GPU跑你的深度学习代码](https://www.jianshu.com/p/000d2a9d36a0)

## TensorFlow Hub

[TENSORFLOW HUB 简介](http://www.rethink.fun/index.php/2018/06/12/tfhub/)

[手把手教你用tensorflow-hub做图像分类（自己的训练数据集）](https://blog.csdn.net/bit_die/article/details/80200022)

[retrain\.py](https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py)

## 参考
