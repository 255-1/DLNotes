+ ***[LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211220083.PNG)

**年份**：  2020

**引用次数**： 428

**应用领域**：  GNN, CF

**方法及优缺点**：

作者发现GCN中最常用的设计, 特征变换和激活函数其实没什么用,所以作者提出了LightGCN只包含最重要的组件邻居消息聚合, 在[NGCF](./NGCF.md)的基础上简化完成

**结论**：

同上, 希望未来能在基于知识图谱和社交网络的前人工作上使用LightGCN完成简化, 以及对于layer合并不同层的权重实现自适应(例如，稀疏的用户可能需要更多来自高阶邻居的信号，而活跃的用户需要更少的信号), 最后会研究是否存在非抽样回归损失的快速解决方案，并将其用于在线工业场景。

**动机**:  

NGCF做特征变换没什么用, 因为对于普通的GCN每个节点有许多属性, 但是CF中的节点只有ID, 没有隐语义能做特征变换.作者通过消融实验发现最有用的就是邻居消息聚合, 所以提出了本文的LightGCN. 本文和SGNN有区别, SGNN是节点分类, 本文是CF,其次SGNN的节点有属性, CF中节点只有ID信息, 并且SGNN效果比LightGCN差

**相关工作和理论**：  

首先作者进行消融分析, 示意图如下, 不同数据库下,-n为去除激活函数,-f为去除特征变换, -fn为全去除, 可以看出全去除对于学习速度和recall都有很大的提升.

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211215522.PNG)

接着作者就提出了LightGCN的设计, 对于最重要的邻居消息聚合函数如下, 前人的GIN, [GraphSAGE](./GraphSAGE.md), BGNN这些聚合都用了激活函数, 其实都不需要

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211218825.PNG" style="zoom:67%;" />

在LightGCN中使用的embedding propagation layer如下, 比起NGCF中去除了匹配度, 去除了特征变换权重矩阵.并且去除了自己到自己的消息传递.

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211221996.PNG" style="zoom:67%;" />

对于最后的embedding整合就是把每一层的embedding加起来, 公式如下, 其中$\alpha_k=\frac{1}{K+1}$, 这个$\alpha$也可以使用变量学, 但是作者说试了没什么改进,使用求和的原因有如下三点

1. 只使用最后一层会有过度平滑的问题,
2. 不同的layer包含不同的user-item交互语义
3. 将不同层的embeding加权和可以捕获具有自连接的图卷积的效果

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211227943.PNG)

最后的预测还是内积相似度$\hat y_{ui}=\bold e_u^T\bold e_i$

作者还给出了矩阵形式的公式如下所示,其中$\tilde A=D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211232991.PNG" style="zoom:67%;" />

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211235478.PNG" style="zoom:67%;" />

优化的loss function依旧选择的BPR,公式如下, 作者希望能用最新的一些负样本抽样策略, 比如hard sampling和对抗样本.除此以外本模型并不像NGCF一样使用dropout, 因为没有特征变换, 只用$L_2$正则化就足够了.

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211604928.PNG)



接着作者就分析了为什么这个有用.首先会讨论基于SGCN论文内容为什么不不需要自连接的消息传递.然后讨论基于APPNP论文中处理过渡平滑的问题是如何解决的, 会举例一个2阶的情况.

首先是对于自连接的情况下,SGCN定义了加了自连接的情况,如下,由于$(D+I)$只是用来重新调整比例所以可以忽略,
$$
E^{(k+1)}=(D+I)^{-\frac{1}{2}}(A+I)(D+I)^{-\frac{1}{2}}E^{(k)}
$$
那么上式可以简写成如下形式,可以看出对于加入自连接等价于不同层的加权求和的结果.

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211444216.PNG" style="zoom:67%;" />

接着对于APPNP是将GCN和PageRank算法组合得到的解决了过度平滑问题, 做法是每一层都将第0层的embedding按照一定比例和后续层相加, 公式如下

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211450935.PNG" style="zoom:67%;" />

最后一层的特征公式如下, 这个之前的$\alpha_k$起相同作用, 通过设定$\alpha$大小就可以控制初始embedding的占比

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211451433.PNG" style="zoom:67%;" />

最后作者举了个2阶的例子, 如下所示(备注),下面这个公式的本质就是对于u找到它的item邻居,在这些item邻居中再找交互过的v

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211542849.PNG" style="zoom:67%;" />

换一种写法如下所示,这个系数就代表v到u的作用.

1. 共同交互过的item越多越好
2. 交互过的item的流行度越少越好
3. v交互过的item数量越少越好.

以上三点正好符合了CF对于用户相似度的要求.

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211547610.PNG" style="zoom:67%;" />

**实验结果**:  

复现回来补

**个人总结**：  

大道至简

**备注**  

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211543143.PNG)