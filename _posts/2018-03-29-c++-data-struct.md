---
layout: post
title: C++中的数据结构
categories: [C++]
description: 介绍一些C++中的数据结构
keywords: C++, 数据结构
---

## vector

如果需要高效的随机存取，而不在乎插入和删除的效率，使用vector。

vector拥有一段连续的内存空间，能很好的支持随机存取。
因此vector\<int\>::iterator支持“+”，“+=”，“<”等操作符。

vector\<int\>::iterator和list\<int\>::iterator都重载了“++”运算符。

一些API：v.push\_back(i)；v[2]；v.begin()；v.end()；v.begin()+1。

[关于C++中vector和set使用sort方法进行排序](https://www.cnblogs.com/zhouxiaosong/p/5557990.html)

## list

如果需要大量的插入和删除，而不关心随机存取，则应使用list。

list的内存空间可以是不连续，它不支持随机访问。
因此list\<int\>::iterator则不支持“+”、“+=”、“<”等。

vector\<int\>::iterator和list\<int\>::iterator都重载了“++”运算符。

## map

[C++中的STL中map用法详解](https://www.cnblogs.com/fnlingnzb-learner/p/5833051.html)

map是一类关联式容器。

```
构造函数: map<int, string> mapStudent
数据的插入: mapStudent.insert(pair<int, String>(1, "student_one"))
            mapStudent.insert(map<int, String>::value_type (1, "student_one"))
            mapStudent[1] = "student_one" 这种方式会覆盖之前关键字对应的值，上述两个方法不可覆盖。
判断数据是否插入成功: pair<map<int, String>::iterator, bool> Insert_Pair
                      Insert_Pair = mapStudent.insert(map<int, String>::value_type (1, "student_one")) 如果插入成功，Insert_Pair.second应该为true。
判断关键字是否出现: mapStudent.count()
		    mapStudent.find()
数据的删除: mapStudent.erase() 通过条目对象/关键字删除。
            mapStudent.clear()
数据的交换: mapStudent.swap() 两个容器中所有元素的交换。
数据的排序: map中的与元素是自动自动按key升序排序。
```

**尤其注意：遍历的时候不应该对map进行删除操作，这样可能会对遍历操作造成混乱。**如下不可取。

```
for (it = vec.begin(); it != vec.end(); it++) {
            if (it->second == 1) {
                vec.erase(it->first); 不可取
	    }
```

## set

```
数据的插入:  insert()
判断元素是否出现: find() 返回一个指向被查找到元素的迭代器
数据的删除: erase()
数据的交换: swap() 交换两个集合变量
数据的排序: set中的元素都会根据元素的键值自动排序。
```

[c++ stl set之顺序排序](https://blog.csdn.net/u014221279/article/details/51172217)

[STL之set的用法](http://www.cnblogs.com/SarahLiu/p/5892900.html)

## stack和queue

push()、pop()、empty()、size()

stack: top()

queue: front()

## Tree

[树的定义、分类及存储（没看）](https://www.cnblogs.com/tarbitrary/p/4030652.html) 

## priority_queue

[C++ priority_queue](https://blog.csdn.net/thearcticocean/article/details/46820883)

## 参考

[map,set,multimap,multiset的异同（没看）](https://blog.csdn.net/qq_36221862/article/details/72808475)

[STL容器之multimap和multiset（没看）](https://blog.csdn.net/ro_wsy/article/details/7698077)

[C++ vector和list的区别](https://www.cnblogs.com/shijingjing07/p/5587719.html)
