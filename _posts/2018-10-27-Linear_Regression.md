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

损失函数为

<a href="https://www.codecogs.com/eqnedit.php?latex=J(W)&space;=&space;\frac{1}{2M}\sum_{i=1}^{M}(h_w(x^i)&space;-&space;y^i)^2&space;=&space;\frac{1}{2M}(XW&space;-&space;Y)^T(XW&space;-&space;Y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(W)&space;=&space;\frac{1}{2M}\sum_{i=1}^{M}(h_w(x^i)&space;-&space;y^i)^2&space;=&space;\frac{1}{2M}(XW&space;-&space;Y)^T(XW&space;-&space;Y)" title="J(W) = \frac{1}{2M}\sum_{i=1}^{M}(h_w(x^i) - y^i)^2 = \frac{1}{2M}(XW - Y)^T(XW - Y)" /></a>

### 算法: 求导/梯度下降

算法即学习模型的具体计算方法。一般问题归结为最优化问题，所以学习的算法即求解最优化问题的算法。

#### 矩阵满秩可求解（求导等于0）

<a href="https://www.codecogs.com/eqnedit.php?latex=L(w)&space;=&space;\frac{1}{2}(XW&space;-&space;Y)^T(XW&space;-&space;Y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(w)&space;=&space;\frac{1}{2}(XW&space;-&space;Y)^T(XW&space;-&space;Y)" title="L(w) = \frac{1}{2}(XW - Y)^T(XW - Y)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=L(W)&space;=&space;\frac{1}{2}(W^TX^TXW&space;-&space;W^TX^TY&space;-&space;Y^TXW&space;&plus;&space;Y^TY)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(W)&space;=&space;\frac{1}{2}(W^TX^TXW&space;-&space;W^TX^TY&space;-&space;Y^TXW&space;&plus;&space;Y^TY)" title="L(W) = \frac{1}{2}(W^TX^TXW - W^TX^TY - Y^TXW + Y^TY)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=L(W)&space;=&space;\frac{1}{2}(W^TX^TXW&space;-&space;2W^TX^TY&space;&plus;&space;Y^TY)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(W)&space;=&space;\frac{1}{2}(W^TX^TXW&space;-&space;2W^TX^TY&space;&plus;&space;Y^TY)" title="L(W) = \frac{1}{2}(W^TX^TXW - 2W^TX^TY + Y^TY)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;L(W)}{W}&space;=&space;2X^TXW&space;-&space;2X^TY" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L(W)}{W}&space;=&space;2X^TXW&space;-&space;2X^TY" title="\frac{\partial L(W)}{W} = 2X^TXW - 2X^TY" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;L(W)}{W}&space;=&space;2X^TXW&space;-&space;2X^TY&space;W&space;=&space;(X^TX)^{-1}X^TY" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L(W)}{W}&space;=&space;2X^TXW&space;-&space;2X^TY&space;W&space;=&space;(X^TX)^{-1}X^TY" title="\frac{\partial L(W)}{W} = 2X^TXW - 2X^TY W = (X^TX)^{-1}X^TY" /></a>

以上的结果只能在<a href="https://www.codecogs.com/eqnedit.php?latex=X^TX" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X^TX" title="X^TX" /></a>**能求逆矩阵**的情况下进行，**如果不是满秩矩阵，不能求逆，我们采用梯度下降的方法更新参数**。

#### 矩阵不满秩（梯度下降）

梯度下降算法是一种求局部最优解的方法。对于某一个函数，某一点的梯度是其增长最快的方向，那么它的相反方向则是该点下降最快的方向。

方法是首先随机初始化参数，然后沿着负梯度方向进行迭代，使得损失函数越来越小。

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;J(W)}{\partial&space;W_j}&space;=&space;\frac{1}{2}\sum_{i=1}^{M}(h_W(x^i)&space;-&space;y^i)x_j^i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;J(W)}{\partial&space;W_j}&space;=&space;\frac{1}{2}\sum_{i=1}^{M}(h_W(x^i)&space;-&space;y^i)x_j^i" title="\frac{\partial J(W)}{\partial W_j} = \frac{1}{2}\sum_{i=1}^{M}(h_W(x^i) - y^i)x_j^i" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=W_j&space;:&space;W_j&space;-&space;\alpha&space;\frac{\partial&space;J(W)}{\partial&space;W_j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W_j&space;:&space;W_j&space;-&space;\alpha&space;\frac{\partial&space;J(W)}{\partial&space;W_j}" title="W_j : W_j - \alpha \frac{\partial J(W)}{\partial W_j}" /></a>

## 一些trick

### 对梯度下降算法的改进

批梯度下降：更新每一参数时都使用所有的样本来进行更新。它得到的是一个全局最优解，但是每迭代一步，都要用到训练集所有的数据。如果样本数目M很大，那么迭代速度会很慢。

随机梯度下降：通过每个样本来迭代更新一次。它伴随的一个问题是噪音较批梯度下降要多，它并不是每次迭代都向着整体最优化方向。

小批量梯度下降：**建议使用的方法**。这种算法的训练过程比较快，而且也能保证最终参数训练的准确率。

我们也可以使用**牛顿法**或者**拟牛顿法**进行求解。拟牛顿法是对二阶导数矩阵进行近似求解。

### 正则化

防止过拟合。L1的目的减少参数，L2的目的使得参数值变少。

L0正则化：模型参数中非零参数的个数。稀疏的参数可以防止过拟合。**但因为L0正则化很难求解，是个NP难问题，因此一般采用L1正则化**。L1正则化是L0正则化的最优凸近似，比L0容易求解，并且也可以实现稀疏的效果。

L1正则化：各个参数绝对值之和。**引入先验，假设参数服从拉普拉斯分布**。

L2正则化：各个参数的平方的和的开方。**引入先验，假设参数服从正态分布**。

### 多项式回归

例如对于自变量的某个分量，增加二次方，就变成了多项式回归的模型，类似于特征组合、特征变换的效果。可以用来扩展特征，**避免欠拟合**。

### 广义线性回归

**多项式回归是对样本特征端做了推广，也可以对输出做推广**。比如我们的输出<a href="https://www.codecogs.com/eqnedit.php?latex=Y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y" title="Y" /></a>不满足和<a href="https://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X" title="X" /></a>的线性关系，但是<a href="https://www.codecogs.com/eqnedit.php?latex=lnY" target="_blank"><img src="https://latex.codecogs.com/gif.latex?lnY" title="lnY" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X" title="X" /></a>满足线性关系，模型函数如下

<a href="https://www.codecogs.com/eqnedit.php?latex=lnY&space;=&space;XW" target="_blank"><img src="https://latex.codecogs.com/gif.latex?lnY&space;=&space;XW" title="lnY = XW" /></a>

对于每个样本的输出<a href="https://www.codecogs.com/eqnedit.php?latex=Y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y" title="Y" /></a>，我们用<a href="https://www.codecogs.com/eqnedit.php?latex=lnY" target="_blank"><img src="https://latex.codecogs.com/gif.latex?lnY" title="lnY" /></a>去对应，从而**仍然可以用线性回归的算法去处理这个问题**。我们把<a href="https://www.codecogs.com/eqnedit.php?latex=lnY" target="_blank"><img src="https://latex.codecogs.com/gif.latex?lnY" title="lnY" /></a>一般化，假设这个函数是单调可微函数<a href="https://www.codecogs.com/eqnedit.php?latex=g(.)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(.)" title="g(.)" /></a>，则一般化的广义线性回归形式是

<a href="https://www.codecogs.com/eqnedit.php?latex=g(Y)&space;=&space;XW" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(Y)&space;=&space;XW" title="g(Y) = XW" /></a>

函数<a href="https://www.codecogs.com/eqnedit.php?latex=g(Y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(Y)" title="g(Y)" /></a>我们通常称为联系函数。

## 线性回归的使用

自变量：离散变量（对于有序的变量，转化为连续值；无序变量可以进行ont-hot处理）、连续变量（可以对连续变量进行适当处理，比如变为0均值归一化、min\_max归一化等）

因变量：连续值。

### 使用条件

适用于回归预测问题。满足如下条件

* 自变量与因变量是否呈**线性关系**。如果因变量Y与某个自变量Xi之间呈现出曲线趋势，可尝试通过变量变换予以修正。常用的变量变换方法有对数变换、倒数变换、平方根变换等。
* 因变量是否符合正态分布。
* 因变量数值之间是否独立。即自变量之间不存在多重共线问题。
* 方差是否齐性。即残差的大小不随所有变量取值水平的改变而改变。

[多重共线性问题的几种解决方法](https://blog.csdn.net/nieson2012/article/details/48980491)

### 优缺点

优点：模型比较简单；可解释性。

缺点：对异常值敏感、容易过拟合、容易陷入局部最优。

## 参考

[机器学习-----线性回归浅谈（Linear Regression）](https://www.cnblogs.com/GuoJiaSheng/p/3928160.html)

[岭回归直接得到最优解的公式推导](https://blog.csdn.net/lw_power/article/details/82953337)

[线性回归推导](http://www.cnblogs.com/hearwind/p/9613297.html)

[线性回归误差项分析](https://blog.csdn.net/qq_35028612/article/details/78632035)

[线性回归原理和实现基本认识](https://blog.csdn.net/lisi1129/article/details/68925799)

[线性回归原理小结](www.cnblogs.com/pinard/p/6004041.html)
