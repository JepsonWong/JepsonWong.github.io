---
layout: post
title: mxnet和pytorch的区别
categories: [深度学习]
description: 总结mxnet和pytorch的一些区别。
keywords: 深度学习, mxnet, pytorch
---

## 自动求导

mxnet若要对一个变量x求导(x = nd.arange(4).reshape((4, 1)))，首先x.attach\_grad()申请存储梯度所需要的内存。为了减少计算和内存开销，默认条件下MXNet**不会记录用于求梯度的计算**，调用autograd.record()来要求MXNet记录与求梯度有关的计算。接下来我们可以通过调用backward()函数自动求梯度。

pytorch：

```
optimizer.zero_grad()   # 清空上一步的残余更新参数值
loss.backward()         # 误差反向传播, 计算参数更新值
optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
```

mxnet梯度不累加；pytorch梯度累加，所以每次需要清空保存的梯度值。

## 模型参数初始化

mxnet定义了模型net = nn.Sequential()，然后调用net.initialize()来初始化模型参数。

pytorch模型定义时就顺便初始化了，不需要专门的初始化步骤。

```
https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
定义Linear层参数初始化过程。
```
