---
layout: post
title: Java : Vector/ArrayList HashTable/HashMap ArrayList/LinkedList
categories: [Java]
description: 介绍一些Java的数据结构
keywords: Java
---

## List : 有序、重复

### ArrayList

底层数据结构是数组，查询快，增删慢。线程不安全，效率高。

### Vector

底层数据结构是数组，查询快，增删慢。线程安全，效率低。

### LinkedList

底层数据结构是链表，查询慢，增删快。线程不安全，效率高。

## Set : 无序、唯一

### HashSet

底层数据结构是哈希表。哈希表依赖两个方法：hashCode()和equals()。

### TreeSet

底层数据结构是红黑树。

## Map : 双列集合

### HashMap

底层数据结构是哈希表。线程安全，效率低。

### HashTable

底层数据结构是哈希表。线程安全，效率低。

### TreeMap

底层数据结构是红黑树。(是一种自平衡的二叉树)

## 底层数据结构的一些对比

### Vector/ArrayList

Vector的方法都是同步的(Synchronized)，是线程安全的(thread-safe)，而ArrayList的方法不是线程安全的。由于线程的同步必然要影响性能，因此，ArrayList的性能比Vector好。

当Vector或ArrayList中的元素超过它的初始大小时，Vector会将它的容量翻倍，而ArrayList只增加50%的大小，这样，ArrayList就有利于节约内存空间。

### HashTable/HashMap

Hashtable和HashMap它们的性能方面的比较类似Vector和ArrayList，比如Hashtable的方法是同步的，而HashMap的不是。

### ArrayList/LinkedList

ArrayList的内部实现是基于内部数组Object[]，所以从概念上讲，它更象数组，但LinkedList的内部实现是基于一组连接的记录，所以，它更象一个链表结构。所以，它们在性能上有很大的差别。

从上面的分析可知，在ArrayList的前面或中间插入数据时，你必须将其后的所有数据相应的后移，这样必然要花费较多时间。所以，当你的操作是在一列数据的后面添加数据而不是在前面或中间，并且需要随机地访问其中的元素时，使用ArrayList会提供比较好的性能；而访问链表中的某个元素时，就必须从链表的一端开始沿着连接方向一个一个元素地去查找，直到找到所需的元素为止。所以，当你的操作是在一列数据的前面或中间添加或删除数据，并且按照顺序访问其中的元素时，就应该使用LinkedList了。

## 参考

[Java中常见数据结构：list与map-底层如何实现](https://blog.csdn.net/xy2953396112/article/details/54891527)

[Vector与ArrayList区别](https://www.cnblogs.com/efforts-will-be-lucky/p/7053666.html)
