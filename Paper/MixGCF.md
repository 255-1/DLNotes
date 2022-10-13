+ ***[MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems](https://dl.acm.org/doi/abs/10.1145/3447548.3467408)***   

![image-20221013193007634](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20221013193007634.png)

**年份**：  KDD 21

**引用次数**： 26

**应用领域**：  对比学习, 协同推荐

**方法及优缺点**：

本文重点研究的是负采样方面的研究, 通过user-item图以及**GNN的消息汇聚过程**中得到负样本, 通过把正样本信息注入到负样本中来得到硬负采样, 挺amazing的

**动机**:  

一般情况使用的是uniform分布进行抽样, [17, 28, 40, 44, 48, 52]进行进一步研究, 再比如PinSage使用PageRank分数进行抽样, 但是这些都只关注了单个图的, 忽视了GNN消息汇聚过程.

负采样公式可以如下形式, 硬负采样可以帮助这个训练, 这在[6,17,30,52]中有学习,但是在GNN推荐系统中没有得到很好的研究,只有PinSage和MCNS[48]有设计而且着重于一个图中的负样本点, 在本文中提出了一种在图的embeding空间中的负样本采样
$$
\max \prod_{v^+,v^-\sim f_S(u)}P_u(v^+>v^-|\Theta)
$$
**相关工作和理论**：  

本文提出了一种基于positive mixing和hop mixing来结合信息的办法来得到负样本.positive mixing通过插值混合方法将正样本的信息注入到负样本中，生成困难负样本候选集合；hop mixing则首先利用一种困难负样本选择策略来从上一步得到的每层困难负样本集合中选择一个唯一的样本，再通过pooling操作结合不同的信息，从而生成假的但富含信息的负样本

![image-20221013190037932](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20221013190037932.png)

### Positive Mixing

先从负样本中采集出M个负样本, 加上层数Layer, 一共有Mx(L+1)个负样本向量, $\mathcal E =\{e_{v_m}^{(l)}\}$𝑚𝑖𝑥𝑢𝑝基于插入的数据增强办法, 把正样本的信息的$\alpha$比例注入到负样本中, $\alpha$为均匀采样(0,1)
$$
{e'}_{v_m}^{(l)}=\alpha^{(l)}{e}_{v^+}^{(l)}+(1-\alpha^{(l)}){e}_{v_m}^{(l)}
$$
将正样本的信息注入到负样本中，从而使得模型更难区分决策边界，增强了模型的识别能力.

通过随机混合系数引入了随机不确定性，从而增强了模型的泛化能力

### Hop Mixing

hop mixing主要使用了GNN分层的思想, 在每层中选取一个负样本向量, 这个不同层的节点可以不同, 将这些不同层的节点特征pool在一起就是我们的负样本
$$
e_{v^-}=f_{pool}({e'}^{(0)}_{v_x},..,{e'}^{(L)}_{v_y})
$$
还有一个问题就是如何选取每层的那一个节点, 这里使用了内积的思想,计算用户向量和负向量间的内积，然后选择最大内积对应负样本, $f_Q(\cdot)$基于你使用的pooling方式, 对于sum的方式$f_Q(u,l)=e_u$对于concat$f_Q(u,l)=e_u^{(l)}$
$$
{e'}^{(l)}_{v_x}=\arg\max\limits_{{e'}^{(l)}_{v_m}\in\mathcal E^{(l)}}f_Q(u,l)\cdot {e'}^{(l)}_{v_m} \\
$$

### Optimization

还是使用的BPR, 只是负采样使用的是上述的方法
$$
\mathcal L_{BPR}=\sum_{(u,v^+)\in\mathcal O^+, e_{v^-}\sim f_{MixGCF(u,v^+)}}\ln\sigma(e_u\cdot e_{v^-}-e_u\cdot e_{v^+})
$$
**实验结果**:  

伪代码如下

![image-20221013192026248](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20221013192026248.png)

**个人总结**：  

**备注**  