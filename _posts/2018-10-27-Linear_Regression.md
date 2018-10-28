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

### 模型: 线性回归函数模型

模型给出了我们要学习的函数形式。

<a href="https://www.codecogs.com/eqnedit.php?latex=h_w(x^i)&space;=&space;w_0&space;&plus;&space;w_1x_1&space;&plus;&space;w_2x_2&space;&plus;&space;...&space;&plus;&space;w_nx_n&space;=&space;W^TX" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_w(x^i)&space;=&space;w_0&space;&plus;&space;w_1x_1&space;&plus;&space;w_2x_2&space;&plus;&space;...&space;&plus;&space;w_nx_n&space;=&space;W^TX" title="h_w(x^i) = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n = W^TX" /></a>

### 策略: 损失函数

策略即考虑按照什么准则学习或者选择最优的模型。

我们这里采用最小二乘法，最小二乘法假设误差（误差即正确值和预测值的差）服从独立正态分布，同时也假设因变量服从正态分布。

何为最小二乘法，其实很简单。我们有很多的给定点，这时候我们需要找出一条线去拟合它，那么我先假设这个线的方程，然后把数据点代入假设的方程得到观测值，求使得实际值与观测值相减的平方和最小的参数。

损失代价函数为

<a href="https://www.codecogs.com/eqnedit.php?latex=J(W)&space;=&space;\frac{1}{2M}\sum_{i=1}^{M}(h_w(x^i)&space;-&space;y^i)^2&space;=&space;\frac{1}{2M}(XW&space;-&space;Y)^T(XW&space;-&space;Y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(W)&space;=&space;\frac{1}{2M}\sum_{i=1}^{M}(h_w(x^i)&space;-&space;y^i)^2&space;=&space;\frac{1}{2M}(XW&space;-&space;Y)^T(XW&space;-&space;Y)" title="J(W) = \frac{1}{2M}\sum_{i=1}^{M}(h_w(x^i) - y^i)^2 = \frac{1}{2M}(XW - Y)^T(XW - Y)" /></a>

### 算法: 求导/梯度下降

算法即学习模型的具体计算方法。一般问题归结为最优化问题，所以学习的算法即求解最优化问题的算法。

#### 矩阵满秩可求解（求导等于0）

<a href="https://www.codecogs.com/eqnedit.php?latex=L(w)&space;=&space;\frac{1}{2}(XW&space;-&space;Y)^T(XW&space;-&space;Y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(w)&space;=&space;\frac{1}{2}(XW&space;-&space;Y)^T(XW&space;-&space;Y)" title="L(w) = \frac{1}{2}(XW - Y)^T(XW - Y)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=L(W)&space;=&space;\frac{1}{2}(W^TX^TXW&space;-&space;W^TX^TY&space;-&space;Y^TXW&space;&plus;&space;Y^TY)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(W)&space;=&space;\frac{1}{2}(W^TX^TXW&space;-&space;W^TX^TY&space;-&space;Y^TXW&space;&plus;&space;Y^TY)" title="L(W) = \frac{1}{2}(W^TX^TXW - W^TX^TY - Y^TXW + Y^TY)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=L(W)&space;=&space;\frac{1}{2}(W^TX^TXW&space;-&space;2W^TX^TY&space;&plus;&space;Y^TY)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(W)&space;=&space;\frac{1}{2}(W^TX^TXW&space;-&space;2W^TX^TY&space;&plus;&space;Y^TY)" title="L(W) = \frac{1}{2}(W^TX^TXW - 2W^TX^TY + Y^TY)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;L(W)}{W}&space;=&space;2X^TXW&space;-&space;2X^TY" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L(W)}{W}&space;=&space;2X^TXW&space;-&space;2X^TY" title="\frac{\partial L(W)}{W} = 2X^TXW - 2X^TY" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;L(W)}{W}&space;=&space;2X^TXW&space;-&space;2X^TY&space;W&space;=&space;(X^TX)^{-1}X^TY" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L(W)}{W}&space;=&space;2X^TXW&space;-&space;2X^TY&space;W&space;=&space;(X^TX)^{-1}X^TY" title="\frac{\partial L(W)}{W} = 2X^TXW - 2X^TY W = (X^TX)^{-1}X^TY" /></a>

以上的结果只能在<a href="https://www.codecogs.com/eqnedit.php?latex=X^TX" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X^TX" title="X^TX" /></a>**能求逆矩阵**的情况下进行，如果不是满秩矩阵，不能求逆，我们采用梯度下降的方法更新参数。

#### 矩阵不满秩（梯度下降）

梯度下降算法是一种求局部最优解的方法。对于某一个函数，某一点的梯度是其增长最快的方向，那么它的相反方向则是该点下降最快的方向。

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;J(W)}{\partial&space;W_j}&space;=&space;\frac{1}{2}\sum_{i=1}^{M}(h_W(x^i)&space;-&space;y^i)x_j^i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;J(W)}{\partial&space;W_j}&space;=&space;\frac{1}{2}\sum_{i=1}^{M}(h_W(x^i)&space;-&space;y^i)x_j^i" title="\frac{\partial J(W)}{\partial W_j} = \frac{1}{2}\sum_{i=1}^{M}(h_W(x^i) - y^i)x_j^i" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=W_j&space;:&space;W_j&space;-&space;\alpha&space;\frac{\partial&space;J(W)}{\partial&space;W_j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W_j&space;:&space;W_j&space;-&space;\alpha&space;\frac{\partial&space;J(W)}{\partial&space;W_j}" title="W_j : W_j - \alpha \frac{\partial J(W)}{\partial W_j}" /></a>

## 一些trick

### 对梯度下降算法的改进

批梯度下降

随机梯度下降

### 正则化

L0正则化

L1正则化

L2正则化
