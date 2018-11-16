---
layout: post
title: Factorization Machines系列
categories: [机器学习]
description: 介绍FM系列模型
keywords: 机器学习, FM, FFM, DeepFM, NFM, AFM
---

## FM

FM在2010年提出，旨在解决**稀疏数据**下的特征组合问题，目前该系列模型在搜索推荐领域被广泛使用。

FM 的优势是可以自动进行特征间的组合，这解决了两个问题：一个是稀疏数据因为特征组合而能得到不错的backgroud model效果；另一个是特征组合解决了人工组合的尴尬。

FM的另一个优点是会**算出每个feature对应的向量v**；这个向量可以看做对feature的一种embeding。

FM 的不足之处是在dense数据集情况下，反而可能表现不佳。

FM公式为：

* 线性模型 + 交叉项
* 交叉项系数为隐向量的内积；将每个特征用k个描述特征的因子来表示；这样二阶交叉项的参数由n * n个较少到n * k个，远少于二阶多项式模型中的参数数量。
* 而且与二阶多项式模型相比，参数学习过程不独立，含有相同特征的参数可以同时用来学习该特征的隐向量。

## FFM

通过引入field的概念，FFM把相同性质的特征归于同一个field。在FFM中，每一维特征xi，针对其它特征的每一种“field”fj，都会学习一个隐向量vi,fj。因此，隐向量不仅与特征相关，也与field相关。

假设每条样本的n个特征属于f个field，那么FFM的二次项有nf个隐向量。而在FM模型中，每一维特征的隐向量只有一个。因此**可以把FM看作是FFM的特例**，即把所有的特征都归属到一个field是的FFM模型。

## Factorisation-machine supported Neural Networks(FNN)

FM参数需要预训练；无法拟合低阶特征；每个field只有一个非零值的强假设。

## Product-based Neural Network(PNN)

对比FNN网络，PNN的区别在于中间多了一层Product Layer层。

## DeepFM

FM考虑了低阶特征的组合问题，但是**无法解决高阶特征的挖掘问题**，所以才引入了DeepFM。

受到Wide&Deep的启发，Huifeng等人将FM和Deep深度学习结合了起来，简单的说就是将Wide部分使用FM来代替，同时FM的隐向量可以充当Feature的Embedding。

[tensorflow源码](https://github.com/ChenglongChen/tensorflow-DeepFM)

和W&D模型相比，DeepFM的wide和deep部分**共享相同的输入**，wide部分使用的是FM，对于组合特征**不再需要手工制作**。用**FM建模low-order的特征组合**，用**DNN建模high-order的特征组合**，因此可以同时从raw feature中**学习到low-和high-order的feature interactions**。在真实应用市场的数据和criteo的数据集上实验验证，DeepFM在CTR预估的计算效率和AUC、LogLoss上超越了现有的模型（LR、FM、FNN、PNN、W&D）。

## 参考

[第09章：深入浅出ML之Factorization家族（有计算复杂度推导）](http://www.52caml.com/head_first_ml/ml-chapter9-factorization-family/)

[深度学习在CTR预估中的应用(好文)](https://baijiahao.baidu.com/s?id=1598859116756700454&wfr=spider&for=pc)

[DeepFM(里面有模型对比图)](https://blog.csdn.net/Liuxz_x/article/details/78949372)

[深入浅出Factorization Machines系列](http://ju.outofmemory.cn/entry/347921)
