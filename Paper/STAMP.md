+ ***[STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation](https://dl.acm.org/doi/10.1145/3219819.3219950)***   

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205071345209.PNG" style="zoom:50%;" />

**年份**：2018  
**引用次数**：279  
**应用领域**：序列推荐  
**方法及优缺点**：

现有的LSTM方法没有考虑用户当前的行为对下一个行为的影响. 作者认为long-term模型可能不足以对长期会话进行建模，这些会话通常包含由非故意点击引起的用户兴趣漂移. 本文的STAMP模型既可以获取用户长session的普遍兴趣, 也考虑了用户最后一次点击所代表的用户兴趣.STAMP强调最后一次点击所反映的用户兴趣.短期兴趣可以在STAMP中得到加强，以便在兴趣变化的情况下准确地捕获用户的当前兴趣，特别是在一个long session中.

**结论**：

本文的两个重要发现:

1. 用户的下一个动作大概率受最后一次点击的影响
2. 本文提出的注意力机制可以有效获取session中的long-term和short-term兴趣.

**动机**:  

RNN模型将session看成是一系列物品进行建模, 并没有明确考虑用户的兴趣随时间变化. 所以用户的短期长期兴趣一样重要. 由此提出本文的模型STAMP,用户的长期兴趣通过external memory(用户在时刻$t$之前的物品embedding序列)来学习, 而用户的短期兴趣也是当前兴趣就是用户的最后一次点击, 注意力机制就是在此之上建立的.

**相关工作和方法**：  

以前的基于兴趣的推荐模型很少考虑会话中不相邻的项目之间的顺序互动，虽然基于一般兴趣的推荐器很好地捕捉了用户的一般品味，但如果不明确地对相邻的联系进行建模，就很难使其推荐适应用户最近的购买行为.

在SWIWO中每个物品在session中的权重使用的固定的方法.而本文的这种引入注意力机制明确之前点击和最后一次点击的相关性. [NARM](./Neural_Attentive_Session-based_Recommendation.md_)将主要意图和session序列特征合并, 将两者当成是同等重要的. STAMP强调最后一次点击所反映的用户兴趣.

接着介绍本文的内容, 作者主要从不带注意力机制的STMP介绍到STAMP模型,首先定义三个向量相似度公式, 就是三个内积
$$
<a, b, c>=\sum_{i=1}^da_ib_ic_i = a^T(b\odot c)
$$
首先介绍STMP, 如下图所示,$m_s$代表当前session的一般兴趣,计算公式为$m_s=\frac{1}{t}\sum_{i=1}^tx_i$(long-term), $m_t$代表用户在session的当前兴趣, $m_t=x_t$(short-term), 然后两个向量通过两个网络结构一样的MLP层进行特征提取. $h_s$的计算公式为$h_s=f(W_sm_s+b_s)$, $f$为tanh, $h_t$计算也一样,然后计算向量相似度$\hat{z}_i=\sigma(<h_s,h_t,x_i>)$这里的$z$可以看成是未归一化的cos相似度,$\sigma$为sigmoid function, 最后计算输出$\hat{y}_i=softmax(\hat{z})$

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205071353008.PNG" style="zoom:50%;" />

STMP的loss function为cross-entropy
$$
\mathcal{L}(\hat{y})=-\sum_{i=1}^{|V|}y_ilog(\hat{y}_i)+(1-y_i)log(1-y_i)
$$
可以看到虽然STMP既计算了long-term和short-term,但是从$m_s$的公式可以看出STMP认为过去的序列是等价的,由此设计出了STAMP.

接着开始介绍STAMP,如下图所示, 主要区别就是加入了一个AttentionNet, 这个attention net主要有两个组件, 一个简单的FNN负责生成注意力权重, 和一个注意力整合生成$m_a$的网络.

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205071345209.PNG" style="zoom:50%;" />

生成注意力权重的FNN公式如下所示, 公式中可以看出特别考虑了短期兴趣$x_t$,这是最大的改进. 此外另一个注意力整合的计算公式很简单$m_a=\sum_{i=1}^t\alpha_ix_i$
$$
\alpha_i=W_o\sigma(W_1x_i+W_2x_t+W_3m_s+b_a)
$$
为了衡量STMP和STAMP的想法, 本文还额外提出了没有Priority的STMO模型,只基于序列的最后一次点击进行预测.STMO的设计就是前面STMP的简化版本,不过多赘述.

**实验结果**:  

复现了回来补

**个人总结**：  

虽然不复杂,不知道作者的STMP模型怎么想到的, 如果前人已经有类似STMP这样的模型,那作者就是加了个Attention并且额外考虑了最后一次点击的注意力分数罢了. 

**备注**  