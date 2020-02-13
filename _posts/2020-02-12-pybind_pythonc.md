---
layout: post
title: Python与C/C++的交互
categories: [Python, C++, C]
description: some word here
keywords: keyword1, keyword2
---

# Pybind和Python C用法

C++不一定比Python运行快，在“起跑”阶段，C++甚至比Python要慢。我们使用C++主要是为了加速大段Python代码。

* 因为Python调用C++需要进行参数转换和返回值转换，这个也会耗费时间。
* Python的\.会搜寻很多东西才能获得到对象的属性、方法等，也会影响执行速度。
* 虽然Python调用C++在类型转换上会有速度损失，但是在进入到函数提内运行过程中的速度是不影响的，假如我们的运算量够大，完全可以弥补那一点点性能影响，所以要想重复利用C++的速度，尽量少调用C++，把计算结果竟然一次性返回，而不是我们多次进行交互，这样就能最大化利用C++。

[如何让你的Python更快](https://blog.zhanglun.me/2018/09/12/%E5%A6%82%E4%BD%95%E8%AE%A9%E4%BD%A0%E7%9A%84Python%E5%83%8FC%E4%B8%80%E6%A0%B7%E5%BF%AB/)
[pybind github](https://github.com/pybind/pybind11)
[pybind11 文档](https://pybind11.readthedocs.io/en/stable/index.html)
[Python3: Python C API参考](https://docs.python.org/3/c-api/index.html)
[Python2: Python C API参考](https://docs.python.org/2/c-api/index.html)

## GIL的获取和释放

当Python端调用C++端的代码时，如果不在C++端主动释放GIL锁，该线程会一直hold GIL锁。

* Pybind用法：py::gil_scoped_release：释放GIL锁；py::gil_scoped_acquire：获取GIL锁
* Python C用法：可以使用Py_BEGIN_ALLOW_THREADS和Py_END_ALLOW_THREADS这一对宏来**释放GIL** [Py_BEGIN_ALLOW_THREADS / Py_END_ALLOW_THREADS](https://docs.python.org/3/c-api/init.html#c.Py_BEGIN_ALLOW_THREADS) ；使用gstate = PyGILState_Ensure()和PyGILState_Release(gstate)来**获取GIL**。

[GIL](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil)

[核对当前C++程序是否占用GIL锁](https://stackoverflow.com/questions/11366556/how-can-i-check-whether-a-thread-currently-holds-the-gil)

[Python之丢掉万恶的GIL](https://juejin.im/entry/5bb094c7e51d450e8b13dff3)

## Python对象

[参考](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/object.html)

py::cast（返回转换后的py::object对象，**是右值，不能对其做取地址操作**）、obj.cast

Python types 包括handle, object, bool\_, int\_, float\_, str, bytes, tuple, list, dict, slice, none, capsule, iterable, iterator, function, buffer, array, and array_t.
* handle: Holds a reference to a Python object (no reference counting)
 * inc_ref(): increase the reference count of the Python object
 * dec_ref(): decrease the reference count of the Python object
* object: Holds a reference to a Python object (with reference counting)
* array: https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html

ref_count(): Return the object’s current reference count.

## Python调用C/C++函数

* 返回值：[返回值策略](https://pybind11.readthedocs.io/en/stable/advanced/functions.html#)
* 增加调用策略：call_guard\<T\>
* 参数：Python object可以直接作为参数（例如上面Python对象中的dict等）；\*args和\*\*kwargs也可以作为参数（py::args起源于py::tuple，py::kwargs起源于py::dict）；通过py::arg定义默认参数

## 智能指针

std::unique_ptr（不能作为参数）、std::shared_ptr [参考](https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#std-shared-ptr)

The binding generator for classes, class\_, can be passed a template type that denotes a special holder type that **is used to manage references to the object**. If no such holder type template argument is given, the default for a type named Type is std::unique_ptr<Type>, which **means that the object is deallocated when Python’s reference count goes to zero**.

## 类

构造函数：
* py::init [参考](https://pybind11.readthedocs.io/en/stable/advanced/classes.html#custom-constructors)
* 还有一种\_\_init\_\_函数。类似如下：
```
py::class_<Raster>(m, "Raster", py::buffer_protocol())
    .def("__init__", [](Raster& raster, py::array_t<double> buffer, double spacingX, double spacingY, double spacingZ) {
    py::buffer_info info = buffer.request();
    new (&raster) Raster3D(static_cast<double*>(info.ptr), info.shape[0], info.shape[1], info.shape[2], spacingX, spacingY, spacingZ);
    })
```

# 参考

其他可以提速Python方法：
* [用 Psyco 让 Python 运行得像 C 一样快](https://www.ibm.com/developerworks/cn/linux/sdk/python/charm-28/index.html)
* [用 Numba 加速 Python 代码，变得像 C++ 一样快](https://zhuanlan.zhihu.com/p/72324090)

相关PR：
* [PR1](https://github.com/PaddlePaddle/Paddle/pull/20983)

