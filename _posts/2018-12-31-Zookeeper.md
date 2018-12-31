---
layout: post
title: ZooKeeper
categories: [分布式]
description: some word here
keywords: ZooKeeper, 分布式
---

## 简介

ZooKeeper是一个分布式的，开放源码的分布式应用程序**协调服务**，是Google的Chubby一个开源的实现，是Hadoop和Hbase的重要组件。它是一个为分布式应用提供**一致性服务**的软件，提供的功能包括：**配置维护、域名服务、分布式同步、组服务**等。 ZooKeeper的目标就是封装好复杂易出错的关键服务，将简单易用的接口和性能高效、功能稳定的系统提供给用户。 ZooKeeper包含一个简单的原语集，提供Java和C的接口。 ZooKeeper代码版本中，提供了分布式独享锁、选举、队列的接口，代码在zookeeper-3.4.3\src\recipes。其中分布锁和队列有Java和C两个版本，选举只有Java版本。

## 重要概念

[分布式学习(1) ---- 初识Zookeeper](https://zhuanlan.zhihu.com/p/24996631)

[可能是全网把 ZooKeeper 概念讲的最清楚的一篇文章](https://zhuanlan.zhihu.com/p/44731983)

* ZooKeeper**本身就是一个分布式程序**（只要半数以上节点存活，ZooKeeper 就能正常服务）。
* 为了保证高可用，最好是以**集群**形态来部署ZooKeeper，这样只要集群中大部分机器是可用的（能够容忍一定的机器故障），那么ZooKeeper本身仍然是可用的。
* ZooKeeper将数据保存在**内存**中，这也就保证了 高吞吐量和低延迟（但是内存限制了能够存储的容量不太大，此限制也是保持znode中存储的数据量较小的进一步原因）。
* ZooKeeper是高性能的。 在“读”多于“写”的应用程序中尤其地高性能，因为“写”会导致所有的服务器间同步状态。（“读”多于“写”是协调服务的典型场景。）
* ZooKeeper有**临时节点**的概念。 当创建临时节点的客户端会话一直保持活动，瞬时节点就一直存在。而当会话终结时，瞬时节点被删除。持久节点是指一旦这个ZNode被创建了，除非主动进行ZNode的移除操作，否则这个ZNode将一直保存在Zookeeper上。
* ZooKeeper底层其实只提供了两个功能：**管理**（存储、读取）用户程序提交的数据；为用户程序提交数据节点**监听**服务。

### Znode

[ZooKeeper的Znode剖析](https://blog.csdn.net/lihao21/article/details/51810395)

在谈到分布式的时候，我们通常说的“节点"是指组成集群的每一台机器。然而，在Zookeeper中，“节点"分为两类，**第一类同样是指构成集群的机器**，我们称之为机器节点；**第二类则是指数据模型中的数据单元**，我们称之为数据节点一一ZNode。

Zookeeper将所有数据存储在内存中，数据模型是一棵树（Znode Tree)，由斜杠（/）的进行分割的路径，就是一个Znode，例如/foo/path1。**每个上都会保存自己的数据内容，同时还会保存一系列属性信息**。

在Zookeeper中，node可以分为持久节点和临时节点两类。所谓持久节点是指一旦这个ZNode被创建了，除非主动进行ZNode的移除操作，否则这个ZNode将一直保存在Zookeeper上。而临时节点就不一样了，它的生命周期和客户端会话绑定，一旦客户端会话失效，那么这个客户端创建的所有临时节点都会被移除。另外，ZooKeeper还允许用户为每个节点添加一个特殊的属性：**SEQUENTIAL**。一旦节点被标记上这个属性，那么在这个节点被创建的时候，Zookeeper会自动在其节点名后面追加上一个整型数字，这个整型数字是一个由父节点维护的自增数字。

### ZooKeeper集群角色

最典型集群模式：Master/Slave模式（主备模式）。在这种模式中，通常Master服务器作为主服务器提供写服务，其他的Slave服务器从服务器通过异步复制的方式获取Master服务器最新的数据提供读服务。

但是，在ZooKeeper中没有选择传统的Master/Slave概念，而是引入了**Leader、Follower和Observer**三种角色。

### ZooKeeper & ZAB协议 & Paxos算法

Paxos算法应该可以说是ZooKeeper的灵魂了。但是，ZooKeeper并没有完全采用Paxos算法，而是使用ZAB协议作为其保证数据一致性的核心算法。另外，在ZooKeeper的官方文档中也指出，ZAB协议并不像Paxos算法那样，是一种通用的分布式一致性算法，它是一种**特别为Zookeeper设计的崩溃可恢复的原子消息广播算法**。

ZAB（ZooKeeper Atomic Broadcast 原子广播）协议是为分布式协调服务ZooKeeper专门设计的一种**支持崩溃恢复的原子广播协议**。 在ZooKeeper中，主要依赖ZAB协议来实现分布式数据一致性，基于该协议，ZooKeeper实现了一种主备模式的系统架构来保持集群中各个副本之间的数据一致性。

ZAB协议两种基本的模式：崩溃恢复和消息广播。

[Zookeeper ZAB 协议分析](http://blog.xiaohansong.com/2016/08/25/zab/)

## ZooKeeper应用

### 统一命名服务

在分布式系统中，经常需要给一个资源生成一个唯一的ID，在没有中心管理结点的情况下生成这个ID并不是一件很容易的事儿。zk就提供了这样一个命名服务。

一般是使用create方法，创建一个自动编号的节点。

### 配置管理

[Zookeeper的功能以及工作原理](https://www.cnblogs.com/felixzh/p/5869212.html)

### 集群管理

管理的是其他集群不是ZooKeeper的集群。

[Zookeeper的功能以及工作原理](https://www.cnblogs.com/felixzh/p/5869212.html)

### 分布式锁

为了确保分布式锁可用，我们至少要确保锁的实现同时满足以下四个条件：

* 互斥性。在任意时刻，**只有一个客户端能持有锁**。
* 不会发生死锁。即使有一个客户端在**持有锁的期间崩溃**而没有主动解锁，也能**保证后续其他客户端能加锁**。
* 具有容错性。只要**大部分的节点正常运行**，客户端就可以加锁和解锁。
* 解铃还须系铃人。**加锁和解锁必须是同一个客户端**，客户端自己不能把别人加的锁给解了。

分布式锁一般有几种实现方式：

* 数据库乐观锁；
* 基于Redis的分布式锁；（基于缓存）
* 基于Memcached的分布式锁；（基于缓存）
* 基于ZooKeeper的分布式锁。

介绍ZooKeeper实现分布式锁：

* 临时节点：客户端可以创建临时节点，当客户端会话终止或超时后Zookeeper会自动删除临时节点。该特性可以用来避免死锁。
* 触发器：当节点的状态发生改变时，Zookeeper会通知监听相应事件的客户端。该特性可以用来实现阻塞等待加锁。
* 有序节点：客户端可以在某个节点下创建子节点，Zookeeper会根据子节点数量自动生成整数序号，类似于数据库的自增主键。

一种比较容易想到的分布式锁实现方案是：

* 检查锁节点是否已经创建，若未创建则尝试创建一个临时节点。
* 若临时节点创建成功说明已成功加锁。若持有锁的客户端崩溃或网络异常无法维持Session，锁节点会被删除不会产生死锁。
* 若临时节点创建失败说明加锁失败，等待加锁。watch锁节点exists事件，当接收到节点被删除的通知后再次尝试加锁。因为Zookeeper中的Watch是一次性的，若再次尝试加锁失败，需要重新设置Watch。
* 操作完成后，删除锁节点释放锁。

该方案存在的问题是，当锁被释放时Zookeeper需要通知大量订阅了该事件的客户端，这种现象称为"惊群现象"或"羊群效应"。

惊群现象对Zookeeper正常提供服务非常不利，因此实践中通常采取另一种方案：

* 创建一个**永久节点作为锁节点**，**试图加锁的客户端**在锁节点下**创建临时顺序节点**。Zookeeper会保证子节点的有序性。
* 若锁节点下**id最小的节点是为当前客户端创建的节点，说明当前客户端成功加锁**。
* 否则**加锁失败**，**订阅上一个顺序节点**。当上一个节点被删除时，当前节点为最小，说明加锁成功。
* 操作完成后，删除锁节点释放锁。
* 该方案**每次锁释放时只需要通知一个客户端**，**避免惊群现象发生**。

该方案的特征是优先排队等待的客户端会先获得锁，这种锁称为公平锁。而锁释放后，所有客户端重新竞争锁的方案称为非公平锁。

[分布式锁的几种实现方式](https://www.cnblogs.com/austinspark-jessylu/p/8043726.html)

[Zookeeper 分布式锁原理](https://blog.csdn.net/li123128/article/details/82827178)

[10分钟看懂！基于Zookeeper的分布式锁](https://blog.csdn.net/qiangcuo6087/article/details/79067136)

[分布式学习(2) ---- Zookeeper实现分布式锁](https://zhuanlan.zhihu.com/p/25010779)

[Redis分布式锁的正确实现方式](https://www.cnblogs.com/linjiqin/p/8003838.html)

[Memcached 和 Redis 分布式锁方案](https://www.cnblogs.com/zrhai/p/4015989.html)

[分布式锁方式（一、基于数据库的分布式锁）](https://blog.csdn.net/tianjiabin123/article/details/72625156)

[Java并发问题--乐观锁与悲观锁以及乐观锁的一种实现方式-CAS](https://www.cnblogs.com/qjjazry/p/6581568.html)

[ABA问题的解决方法](https://blog.csdn.net/lm1060891265/article/details/81747510)

## 参考

