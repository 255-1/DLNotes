+ ***[Rethinking the Item Order in Session-based Recommendation with Graph Neural Networks(FGNN)](https://dl.acm.org/doi/10.1145/3357384.3358010)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206121707747.PNG)

**年份**： 2019

**引用次数**： 94

**应用领域**：  GNN, 序列推荐

**方法及优缺点**：

和[SRGNN](./SRGNN.md)一样都是将序列转换为序列图, 但是FGNN是将一个序列转换成一个图, 不同序列之间的图独立, 然后使用Readout操作得到序列图的特征, 将序列推荐问题转成了图分类问题.  除此以外作者提出了weighted attention graph layer来更新节点embedding.

**结论**：

同上

**动机**:

GRU4Rec, NARM这些模型都认为用户偏好是随着时间变换的, 但这种模式太简单无法捕获复杂的兴趣转换. 对于SRGNN学习到的item embedding后只对最后一个点击物品做自注意力模型用以提取session-level feature, 它忽视了session中特定物品转换模式.  

所以提出了本文的FGNN模型, 通过WGAT学习每个序列图内部的转换模式, 再通过readout学习整个序列的特征

**相关工作和理论**：

作者分为四个部分介绍, Session Graph, WGAT, ReadOut, Recommendation

 首先介绍**序列图生成**.在SRGNN的基础上, 对没有自环的节点添加自环, 并且边也有了权重, 权重就是出现的频率, 添加自环的权重为1.  示意图如下

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206121753136.PNG" style="zoom:50%;" />

接着介绍**WGAT**, GAT一般是在无权重的无向图上使用, 作者将它用在了有向的序列图上并且考虑了边的权重提出了WGAT. one-hot编码的物品通过一个embedding layer得到初始化向量$x_i^0=Embed(v_i)$, 然后对于节点j对当前节点i的注意力系数$e_{ij}$计算公式如下, 这个节点j可以是所有的node节点, 但是计算量太大, 这里只计算邻居节点的注意力系数, 可以通过堆叠多个WGAT Layer学习到k-hop邻居的信息.
$$
e_{ij}=Att(Wx_i, Wx_j, w_{ij})
$$
对于所有的邻居的注意力分数通过softmax算出比例, 公式如下
$$
\alpha_{ij}=softmax(e_{ij})=\frac{exp(e_{ij})}{\sum_{k\in \mathcal N(i)}exp(e_{ik})}
$$
公式(1)中的Attention函数可以是各种各样的, 本模型使用一层MLP和LeakyReLU, 类似GAT中的设置,公式如下
$$
\alpha_{ij}=\frac{exp(LeakyReLu(W_{att}[Wx_i||Wx_j||w_{ij}]))}{\sum_{k\in \mathcal N(i)}exp(LeakyReLu(W_{att}[Wx_i||Wx_k||w_{ik}]))}
$$
最后通过得到的注意力系数$\alpha$更新节点$x_i'=\sigma(\sum_{j\in\mathcal N(i)}\alpha_{ij}Wx_j)$, 根据前人的经验, 使用多头注意力可以稳定训练过程, 多头注意力公式如下

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206121812077.PNG)

但是上述的级联会让向量维度很大, 为了统一维度所以对多头求平均, 最后的公式如下

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206121813271.PNG)

然后作者介绍**ReadOut操作**, Readout函数需要学习item transition pattern的顺序，避免时间顺序的偏差和self-attention对最后一个输入item的不准确, 对于readout操作最需要的是变换不变性. 所以作者调整Set2Set, 公式如下, 具体细节有待学习, 但是通过GRU和自注意力机制, 得到了一个图的特征$q^*_t$

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206121821494.PNG)

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206121821479.PNG)

最后作者介绍**Recommendation模块**,公式如下,将readout得到的图特征作为session的特征,经过一个$W_{out}$的转换和item embedding做内积求相似度, $\hat z$得到的是对每一个item的相似度分数
$$
\hat z = (W_{out}q^*_t)^TX^0
$$
将上述的分数通过softmax转换成概率分布$\hat y=softmax(\hat z)$ 对Top-K物品进行推荐. loss function定义如下,将label的物品通过one-hot编码和$\hat y$进行交叉熵损失计算.
$$
L = -\sum_{i=1}^l one-hot(v_{label,i})log(\hat {y_i})
$$
**实验结果**:  

复现回来补

**个人总结**：  

**备注**  