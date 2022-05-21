+ ***[Neural Graph Collaborative Filtering](https://dl.acm.org/doi/abs/10.1145/3331184.3331267)***   

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211005793.PNG" style="zoom:67%;" />

**年份**：  2019

**引用次数**： 676

**应用领域**：  GNN, CF Recommendation

**方法及优缺点**：

学习用户/物品的特征是非常重要的一件事情, 现在的方法都是通过学习物品ID和属性的办法获得, 但是没有考虑用户和物品交互之间的关系, 作者将这种关系叫做collaborative signal. 本文通过学习物品-用户交互的二部图. 主要贡献就是把GNN和CF合在一起.

**结论**：

同上并且展望了以下未来, 比如加入attention机制,类似GAT. 本文代表了在CF中利用结构性的消息传递机制, 还有其他的信息可以提高对理解用户行为, 比如交叉特征, 知识图谱, 社交网络.

**动机**:  

CF的前提假设是认为相似的用户对于物品的偏好是一样的.之前的基于ML和DL的方法没有考虑交互信号, 但是实际中交互的信号可能达到一个非常大的数值.所以在本模型中需要处理高阶关联(high-order connectivity). 示意图如下, 左边为交互图, 右边为基于$u_1$的高阶关联, 这里的高阶关联是示意图, 在实际中是通过embedding function通过递归转播完成而不是生成树. 这些function叫做embedding propagation layer, 通过堆叠layer就可以得到想要阶数的关联, 叠2层就是$u_2\to i_2\to u_1$

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205210952358.PNG" style="zoom:67%;" />

**相关工作和理论**：  

主要由三个模块

1. embedding layer 提供用户和物品的特征初始化
2. 多层 embedding propagation layer提取用户的高阶连接
3. 预测层, 提取不同阶数中的embedding输出user-item匹配度

首先介绍embedding layer, 如下所示, 在传统的CF算法中会直接计算user-item的分数得到预测匹配度, 但是在NGCF中会从user-item交互图中提取不同阶数的embedding.

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211010958.PNG" style="zoom:67%;" />

接着介绍embedding propagation layers, 这里的实现是基于消息传递机制完成的, 先介绍一阶的情况再推广到高阶情况. 处理过程主要分为消息生成和消息汇聚, 和GNN的思想一致.

首先是消息生成, 所谓消息的定义如下, $m_{u \gets i}$就是需要传播的信号, $f(\cdot)$是消息编码函数, $p_{ui}$是当作衰减因子,所谓衰减因子就是反应item对当前user的共享程度.
$$
\bold m_{u \gets i}=f(\bold e_i, \bold e_u, p_{ui})
$$
再本模型中的具体设计如下, 不仅考虑了物品$i$的权重, 还考虑了$e_i$和$e_u$的匹配度, 其中衰减因子同[Semi-GCN](./Semi_GCN.md)中$L=D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$
$$
\bold m_{u\gets i}=\frac{1}{\sqrt{|\mathcal N_u||\mathcal N_i|}}(\bold W_1\bold e_i+W_2(\bold e_i \odot e_u))
$$

接着是介绍消息汇聚过程, 刚刚把周围节点的消息都生成出来了, 现在需要把周围的消息都汇聚起来, 公式如下, 周围节点的信息求和然后加上自生的信息(和上面的$\bold W_1$)共享参数. 同理对于$m_{i\gets u}$也差不多
$$
\bold e_u^{(1)}=LeakyReLU(\bold m_{u \gets u}+\sum_{i \in\mathcal N_u}\bold m_{u \gets i})\\
m_{u\gets u}=\bold W_1\bold e_u
$$
然后就是把一阶情况推广到高阶,
$$
\bold e_u^{(l)}=LeakyReLU(\bold m_{u \gets u}^{(l)}+\sum_{i \in\mathcal N_u}\bold m^{(l)}_{u \gets i})\\
m^{(l)}_{u\gets u}=\bold W_1^{(l)}\bold e_u^{(l-1)}\\
\bold m^{(l)}_{u\gets i}=p_{ui}(\bold W_1^{(l)}\bold e^{(l-1)}_i+W^{(l)}_2(\bold e^{(l-1)}_i \odot e^{(l-1)}_u))
$$
示意图如下,可以看到在3-hop下$e_{i4}$和$e_{u2}$的交互关系有embedding到特征中.

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211110364.PNG" style="zoom:67%;" />

为了简化实现,将传播过程使用矩阵实现的公式如下

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211113461.PNG" style="zoom:67%;" />

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211114119.PNG" style="zoom:67%;" />

最后是预测层, 经过$L$层layer之后, 每一层的embedding都有那一阶的交互情况,所以这里使用的concate合并所有的embedding, 公式如下. 对于不同的顺序假设也可以使用其他方法聚合比如max pooling, LSTM, average等.这里使用concate因为不需要学习新的变量.

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211118422.PNG" style="zoom:67%;" />

最后产生user-item的匹配度, 使用的内积相似度.

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211120577.PNG" style="zoom:67%;" />

最后作者讨论一下优化的loss function, 使用BPR loss,如下所示, 其中$\Theta$为所有变量.为了防止overfitting, 使用了message和node dropout. 

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205211129199.PNG)

**实验结果**:  

复现回来补

**个人总结**：  

**备注**  