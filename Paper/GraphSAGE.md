+ ***[Inductive Representation Learning on Large Graphs(GraphSAGE)](https://arxiv.org/abs/1706.02216)***   

**年份**：2017  

**引用次数**：4960 

**应用领域**：GNN  

**方法及优缺点**：

[GCN](./GCN.md)这种办法处理不了动态图, 训练时就需要包含全部的节点, 对于unseen节点需要重新训练, 所以提出了本文的GraphSAGE, 一种inductive learning的方法, 本模型可以通过sampling和aggregating节点周围的邻居节点学习未见过的节点的feature.

**结论**：

在同上的基础上展望味蕾, 希望扩展到有向图或者多模态上, 另一方面的工作可以让sampling邻居节点的数量不被固定, 甚至直接学习sampling function完成这件事情.

**动机**:  

还是同样的, [GCN](./GCN.md)这种方法处理不了动态图, 但是现实业务很多都会遇到没见过的顶点, 这时候一种inductive的学习方法显得尤为重要, 甚至你可以在学习蛋白质图结构的基础上直接将feature应用于其他类似的图结构数据上.这种inductive learning要学到节点和邻居直接的结构信息, 这种结构信息既要学到local的也要学到global的. 所以本文在GCN的基础上进行改进使用了一些trainable aggregation functions.

**相关工作和理论**：  

GraphSAGE主要完成以下几个目标

1. 一些trainable aggregators, 不同func处理不同hop的邻居.
2. 设计了无监督的loss让模型不受限于task
3. 模型训练完成后生成新节点的feature

首先作者先介绍结果3, 就是在训练完模型的情况下, 面对新进入的节点如何学习feature, 算法公式如下所示, 对于新节点$x_v$ 通过训练好的一群aggregator函数整合周围邻居信息, 然后和原本特征concate一起做一个fully connected和归一化, 最后将feature简写就是$z_v$

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205171741954.PNG" style="zoom:50%;" />

上述的办法中的邻居数量是固定的,为了让每个batch中间的计算量一样.  作者最好的实践结果是K=2因为每一层都能汇聚2-hop邻居其实层数高了也就全获得了, 其次是邻居数量选择也是个超参数一般两次扩展的$S_1\cdot S_2\leq 500$就很好了

接着作者介绍GraphSAGE是如何学习的2, 因为是要无监督学习, 所以loss func希望图中相近的节点的特征值更相似, 不同的节点之间特征值不同.公式如下, $\bold z_v$为选择到的邻居,送入这个损失函数的$\bold z_u$是由一个节点的局部邻域所包含的特征生成的, 而不是为每个节点训练一个独特的嵌入, 这是和之前的[GCN](./GCN.md)最大的不同.
$$
J_\mathcal{G}(\bold z_u)=-\log(\sigma(\bold z_u^T\bold z_v))-Q\cdot\mathbb E_{v_n\backsim P_n(v)}\log(\sigma(-\bold z_u^T\bold z_{v_n}))
$$
最后作者介绍1中的这些aggregator的设计, 对于图的周围节点是没有顺序关系的, 最理想的情况这个aggregator是对称的, 对输入的排列组合后的结果是一致的.为此作者提出了三个aggregator:

1.GCN Aggregator 如下所示, 就是求和平均, 个人感觉是这个公式不是AGGREGATE,而是直接替换了算法1中4,5两行

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205171828106.PNG" style="zoom:50%;" />

2.LSTM Aggregator, 将选出来的节点信息排列成序列送进LSTM, 神秘做法, 而且结果不对称.

3.Pooling Aggregator, 公式如下, 把各个邻居节点单独经过一个MLP得到一个向量, 然后用element-wise 的max或者mean-pooling(二者效果没差别), 这里的MLP是为了学习到周围点的特征的函数表示, 使用max pooling能有效捕获不同的aspects, 因为最后周围的feature学出来是不同的向量, element-wise的max pooling相当于是把不同邻居向量同一维度下最大的抽出来,类似全取最优这种思想,原则上任何对称的max operator都可以用在这里.

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205171835775.PNG" style="zoom:55%;" />

**实验结果**:  

复现回来补

**个人总结**：

GNN了解差不多了, 先告一段落, 开始看GNN+RecSys了  

**备注**  