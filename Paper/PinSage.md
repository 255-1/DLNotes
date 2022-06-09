+ ***[Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://dl.acm.org/doi/abs/10.1145/3219819.3219890)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206091553617.PNG)

**年份**：  2018

**引用次数**： 1504

**应用领域**：  GNN

**方法及优缺点**：

工业上对大规模数据集实现GCN的方法30亿节点,180亿条边,75亿个样本,本文结合了随机游走和图卷积生成了节点的embedding, 并且作者使用了hard negative sample提升鲁棒性并取得到不错的效果, 除此以外作者通过importance pooling和curriculum training 提高了embedding的效果.

**结论**：

同上, 证明了GCN在大规模推荐系统的可行性.

**动机**:  

GCN能学习到图结构的信息和节点内容的信息, 但是想要扩展GCN到数十亿的节点是一大挑战.所以工业化所需要实现的技术有如下几点:

1. 超快的图卷积, 本模型通过对节点周围的邻域进行采样并从该采样的邻域动态构建计算图来执行高效的局部卷积
2. 生产者-消费者的minibatch生成方式(工业)
3. MapReduce方法减少重复计算(工业)

在算法层面,实现以下几点:

1. 通过随机游走生成卷积, 如果需要计算所有邻居节点会带来巨大的计算开销, 所以使用了随机游走生成计算图
2. Importance pooling, 引入了一种基于随机游走相似性度量来衡量节点特征在此聚合中的重要性的方法
3. curriculum training, 使用了hard negative sample方法学习

本模型和[GraphSAGE](./GraphSAGE.md)紧密相关, 作者调整模型以避免对整个图拉普拉斯算子进行操作,通过消除整个图存储在 GPU 内存中的限制，从根本上改进了 GraphSAGE，使用低延迟随机游走对生产者-消费者架构中的图邻域进行采样.作者还引入了一些新的训练技术来提高性能和 MapReduce 推理管道，以扩展到具有数十亿节点的图形。

**相关工作和理论**：  

本模型应用了多个卷积模块, 每个模块都学习如何从一个小的图邻域聚合信息. 通过堆叠多个模块可以获得本地的拓扑信息. 最重要的是这些卷积参数在节点之间共享, 所以与输入图的大小无关.

首先作者介绍了卷积算法, 伪代码如下, 

Line1:信息汇聚, 邻居节点通过一个DNN然后通过一个aggregate/pooling的$y$

Line2: 信息更新, 级联当前的embedding和邻居节点的embedding

Line3: 归一化稳定训练.

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206091712314.PNG)

接着作者介绍了怎么选取邻居节点, 从节点$u$开始随机游走并计算随机游走访问的节点的L1归一化访问计数, 最高的top T为需要计算的邻居节点. 这种基于重要性的邻域定义的优点有两个: 

1. 选择固定数量的节点进行聚合允许我们在训练期间控制算法的内存占用
2.  其次，它允许算法 1 在聚合邻居的向量表示时考虑邻居的重要性

将上述的卷积模块堆叠起来, 伪代码见备注.

然后作者介绍模型训练相关内容, 笔记就记录算法方面的, 从loss function到 负样本采样

loss function如下所示, $z_{nk}$为负样本, 如果负样本的相似度-和正样本相似度的差超出一个阈值$\Delta$,则loss不为0, 否则为0.

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206091738730.PNG)

接着是负样本采样,为了提高大批量训练的效率，作者抽样了一组 500个的负样本，供每个小批量中的所有训练示例共享, 但是500个在数据集中数量太少了, 要分辨的话很简单但是在总数据集中并不精确, 所以提出了hard negative sample. 具体来说就是将query item通过Personalized PageRank scores排名图中的所有物品, 排名2000-5000作为hard-sample, 前1000为推荐列表中出现的, 为了让模型学习到更加细粒度的区别. 除此以外为了快速收敛, 第一个epoch不使用hard negative sample, 为了学习到大致的变量空间, 后面再引入, 并且引入的方式是在第n个epoch就添加n-1个hard negative sample的方法.

除了上述的算法方面的改进, 在工业化中也额外使用了许多办法比如通过MapReduce避免重复计算, 使用Efficient nearest-neighbor查询进行推荐, 多GPU进行训练, 使用生产者-消费者生成计算图,将内存和GPU数据打通等操作, 需要再回来补

**实验结果**:  

复现再回来补.

**个人总结**：  

工业化的推荐系统就是不一样, 算法方面其实和GraphSAGE差不多, 召回层的负样本采样被着重介绍, 除此以外为了处理大规模数据又有了一系列操作. 

**备注**  

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206091732065.png)