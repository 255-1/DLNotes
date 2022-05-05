+ ***[Neural Attentive Session-based Recommendation](https://arxiv.org/abs/1711.04725v1)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205051431748.PNG)

**年份**：2017  
**引用次数**：539  
**应用领域**：序列推荐, 注意力机制  
**方法及优缺点**：  

前人在序列推荐中只考虑了整个序列特征, 本文在之前的基础上使用注意力机制额外整合了用户的意图(NARM).  

**结论**：  

本文的NARM作为encoder-decoder架构,不仅考虑了序列的特征还考虑了用户对物品的主要意图. 在两个流行的benchmark中都取得了很好的效果. 作者展望了未来跟多的的物品属性能进一步加强效果.  

**动机**:

前人的工作 ,比如GRU4Rec都只考虑了序列的特征而没有考虑用户的主要意图, 当用户出于好奇意外点击其他不相关的物品的时候, 模型效果并不好,  所以作者认为用户的意图也是需要一起考虑的, 由此提出了本文的NARM, 在encoder中整合进了注意力机制, 并且在decoder中将学习到的特征和涉及到的物品做bi-linear匹配

**相关工作**： 

 简单介绍了一下推荐系统中用过的方法, 首先传统的推荐系统可以分为普通推荐和序列推荐, 普通推荐主要是基于item2item的方法, 这些传统的方法都只考虑了最后一次点击, 而序列推荐则是基于Markov链, 考虑了之前的多次点击比如MDP, 但是基于MC的一个问题是他的state空间很快塌陷. 然后就是介绍了基于深度学习的方法, 有前人的基于玻尔兹曼有限机的CF的办法, 再比如GRU4Rec的办法.

本文的数据流如下图所示,x通过encoder得到隐特征h, 并且和注意力信号$\alpha_t$ 整合在一起得到当前时刻的特征并且把这个送入decoder中, 对矩阵$U$做transformed, 由此得到$y$ 

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205051440710.PNG)

作者的序列特征通过global encoder生成, 用户意图通过local encoder生成.

首先介绍global encoder, 如下图所示,和[GRU4Rec](./Session-Based_Recommendations_with_recurrent_neural_networks.md)的用法一致.不过多赘述.文中出现的公式也就是GRU的基本公式

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205051452069.PNG)

接着介绍local encoder, 如下图所示,依旧使用GRU为底层组件, 为了获取用户意图, 作者使用了item-level的注意力机制 $c^l_t=\sum^t_{j=1}\alpha_{tj}h_j$ 其中$\alpha_{tj}=q(h_t, h_j)$为注意力分数, 其中$q$用来计算当前的$h_t$和之前点击的$h_j$, $q(h_t, h_j) = v^T\sigma(A_1h_t+A_2h_j)$, 这里的内容和自注意力的内容差不多, 不难理解.

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205051455599.PNG)

最后将两个encoder的特征concate在一起得到$c_t=[c_t^g;c_t^l]=[h^g_t;\sum_{j=1}^t\alpha_{tj}h_t^l]$ 接着就需要decode了, 但是如果使用fully-connected later会涉及到所有的物品, 计算代价非常大, 常用降低计算复杂度的办法有hierachical softax和随机负样本采样, 但是作者在这里使用了bi-linear decoding scheme, $S_i=emb_i^TBc_t$, 并且在学习过程中并不使用GRU4Rec中的mini-batch并行, 由于需要使用注意力机制所以需要单独计算每个序列. 最后使用的是交叉熵损失函数和Back-Propagation Through Time对序列进行反向传播.其中交叉熵损失为$L(p,q)=-\sum_{i=1}^mp_ilog(q_i)$q为预测概率p为正确分布.

**实验结果**:  

复现回来补

**个人总结**：

不难, 在GRU的基础上加上了自注意力机制来获取用户意图.  
**备注**  