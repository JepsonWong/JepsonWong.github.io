---
layout: post
title: Python与C/C++的交互
categories: [Python, C++, C]
description: some word here
keywords: keyword1, keyword2
---

# Pybind用法

C++不一定比Python运行快，在“起跑”阶段，C++甚至比Python要慢。我们使用C++主要是为了加速大段Python代码。

* 因为Python调用C++需要进行参数转换和返回值转换，这个也会耗费时间。
* Python的\.会搜寻很多东西才能获得到对象的属性、方法等，也会影响执行速度。
* 虽然Python调用C++在类型转换上会有速度损失，但是在进入到函数提内运行过程中的速度是不影响的，假如我们的运算量够大，完全可以弥补那一点点性能影响，所以要想重复利用C++的速度，尽量少调用C++，把计算结果竟然一次性返回，而不是我们多次进行交互，这样就能最大化利用C++。

## GIL的获取和释放

gil_scoped_release

gil_scoped_acquire

[GIL](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil)


