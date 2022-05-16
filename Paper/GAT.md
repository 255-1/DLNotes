+ ***[GRAPH ATTENTION NETWORKS](https://arxiv.org/abs/1710.10903)***   

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205161725096.PNG" style="zoom:50%;" />

**年份**：  2017

**引用次数**： 5072

**应用领域**：  GNN

**方法及优缺点**：

在图结构数据上使用self-attention机制, 能自己学习当前节点和周围节点的权重. 并且本文解决了spectral graph的几个关键问题, 使得GAT模型可以应用于inductive和transductive问题.

**结论**：

本文使用的self-attention机制计算十分高效, 可以并行计算所有节点, 这种办法解决了GCN的缺点, 也不需要对图的先验知识.作者展望了一些未来工作,比如解决更大的batch-size, 利用注意力机制进行模型的可解释性问题, 将节点分类问题拓展到图分类问题, 或者是将边信息也纳入到模型中等.

**动机**:  

对于spectral图的一个问题就是动态图不能计算, 如果图中增加了一个节点就需要重新计算图中所有节点的feature, 对于spatial图的问题是, 如何定义邻居之间的计算, 可以处理不同数量的邻居,并且能保持CNN中卷积的参数共享的问题.GAT和前人工作的不同之处有

- 计算高效, GAT花费$O(|V|FF'+|E|F)$, on par with GCN
- 对比GCN, 即使是相同的邻居,GAT可以分配不同的注意力机制
- 注意力机制不依赖图结构, 解决了GCN中要用无向图邻接矩阵表示的限制, 可以是有向图, 并且可以解决动态图问题  
- 对于GraphSAGE中限制邻居的数量, 而我们不限制
- 和MoNet相比, GAT使用的是节点的feature做相似度而非节点的结构属性

**相关工作和理论**： 

首先介绍Attention Layer的操作

假设每层的输入为$F$输出为$F'$,为了将每层输入的特征向量映射到高层的空间进行自注意力训练, 所以要做一个tranform, 公式如下, 其中$W\in \mathbb R^{F'\times F}$, 其中$a$为一个相似度, 这中计算能让节点$i$和图中所有其他节点$j$都做计算, 丢失了结构信息, 所以这里对这个做一点改进, 只计算当前节点和邻居之间的相似度.
$$
e_{ij}=a(W\vec h_i, W\vec h_j)
$$
上面得到了注意力分数了,那由于有不同数量的邻居,就要将周围的节点的信息整合归一, 公式如下,得到了最后的注意力数值.
$$
\alpha_{ij}=softmax_j(e_{ij})=\frac{exp(e_{ij})}{\sum_{k\in \mathcal N_i}exp(e_{ik})}
$$
在具体实验中, 作者对相似度使用的是一个一层的FFN: $\vec a\in \mathbb R^{2F'}$, 激活函数用的LeakyReLU, 所以本文的注意力公式如下,就是替换了对应位置的变量. 
$$
\alpha_{ij}=\frac{exp(LeakyReLU(\vec a^T[W\vec h_i||W\vec h_j]))}{\sum_{k\in \mathcal N_i}exp(LeakyReLU(\vec a^T[W\vec h_i||W\vec h_k]))}
$$
上面得到了标准化后的注意力数值后就可以将新feature用旧feature线性表示, 公式如下
$$
\vec h'_i=\sigma(\sum_{j\in \mathcal N_i}\alpha_{ij}W\vec h_j)
$$
为了稳定学习过程(感觉是编的), 使用了多头注意力机制, 在上述的方法加入了$K$个注意力头公式如下
$$
\vec h'_i=\overset{K}{||} \sigma(\sum_{j\in \mathcal N_i}\alpha_{ij}^kW^k\vec h_j)
$$
上面这张concate多个注意力头的办法在最后一层中效果并不好, 所以在最后一层使用average办法并且延迟使用一个激活函数收尾, 公式如下
$$
\vec h'_i=\sigma(\frac{1}{K}\sum_{k=1}^K\sum_{j\in \mathcal N_i}\alpha_{ij}^kW^k\vec h_j)
$$


作者对GAT在实现上的限制在Section 2.2最后提了一下, 如果有需要再进一步研究这一块的内容.

**实验结果**:  

复现回来补

**个人总结**：  

**备注**  