---
layout: post
title: 线性回归
categories: [机器学习]
description: some word here
keywords: 机器学习
---

## 线性模型

线性模型的基本形式是：给定由<a href="https://www.codecogs.com/eqnedit.php?latex=d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d" title="d" /></a>个属性描述的示例<a href="https://www.codecogs.com/eqnedit.php?latex=x&space;=&space;(x_1;&space;x_2;&space;...;&space;x_d)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x&space;=&space;(x_1;&space;x_2;&space;...;&space;x_d)" title="x = (x_1; x_2; ...; x_d)" /></a>，其中<a href="https://www.codecogs.com/eqnedit.php?latex=x_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i" title="x_i" /></a>是<a href="https://www.codecogs.com/eqnedit.php?latex=x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x" title="x" /></a>在第<a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a>个属性上的取值，线性模型试图学得一个通过属性的线性组合来进行预测的函数，即

<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;w_1x_1&space;&plus;&space;w_2x_2&space;&plus;&space;...&space;&plus;&space;w_dx_d&space;&plus;&space;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;w_1x_1&space;&plus;&space;w_2x_2&space;&plus;&space;...&space;&plus;&space;w_dx_d&space;&plus;&space;b" title="f(x) = w_1x_1 + w_2x_2 + ... + w_dx_d + b" /></a>

一般用向量形式写成

<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;w^Tx&space;&plus;&space;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;w^Tx&space;&plus;&space;b" title="f(x) = w^Tx + b" /></a>

其中<a href="https://www.codecogs.com/eqnedit.php?latex=w&space;=&space;(w_1;&space;w_2;&space;...;&space;w_d)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w&space;=&space;(w_1;&space;w_2;&space;...;&space;w_d)" title="w = (w_1; w_2; ...; w_d)" /></a>，学得<a href="https://www.codecogs.com/eqnedit.php?latex=w" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w" title="w" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b" title="b" /></a>之后，模型就得以确定。

线性模型形式比较简单，但蕴含着机器学习中一些比较重要的思想。许多非线性模型可以在线性模型的基础上通过引入层级结构或者高维映射得到。而且<a href="https://www.codecogs.com/eqnedit.php?latex=w" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w" title="w" /></a>直观表达了各个属性在预测中的重要性，因此具有**可解释型**。

线性回归即是一种比较简单的线性模型。

## 线性回归

### 线性回归函数模型

<a href="https://www.codecogs.com/eqnedit.php?latex=h_w(x^i)&space;=&space;w_0&space;&plus;&space;w_1x_1&space;&plus;&space;w_2x_2&space;&plus;&space;...&space;&plus;&space;w_nx_n&space;=&space;W^TX" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_w(x^i)&space;=&space;w_0&space;&plus;&space;w_1x_1&space;&plus;&space;w_2x_2&space;&plus;&space;...&space;&plus;&space;w_nx_n&space;=&space;W^TX" title="h_w(x^i) = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n = W^TX" /></a>

### 
