+ ***[Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)***   

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205061429734.PNG" style="zoom:50%;" />

**年份**：2018  
**引用次数**：492  
**应用领域**：自注意力推荐系统  
**方法及优缺点**：

为了获取用户行为的上下文context信息, 有两大类的办法被提出, 第一种是马尔可夫链的方式(MCs), 在处理稀疏数据上效果比较好, 第二种是基于RNN的, 在大量数据上有更好的效果, 本文的目标SASRec是平衡两种目标, 使用自注意力机制让我们既能像RNN一样捕获到长期的语义, 也能像MCs一样基于过去的部分活动. 在SASRec中想要找到和用户历史行为相关的物品, 用于预测下一次点击的物品.

**结论**：

本文提出的SASRec模型作为next-item recommendation, 对用户序列进行建模, 而不使用任何的RNN, CNN的办法, 并且最后在稀疏或者稠密数据上都取得了很不错的效果, 在未来希望能加入更多的上下文信息,并且能够处理超长序列数据.

**动机**:  

MCs适合处理稀疏数据, RNN适合处理稠密数据, 而且由于受到基于自注意力的Transformer的启发. 所以提出本文的模型, 希望它能处理上面的两个问题, 既能像RNN一样处理过去的所有数据, 也可以像MCs一样着重其中某几个行为.

**相关工作**：  

符号定义如下图所示,本文主要描述三个模块, 主要目标就是要$(S_1^u,S_2^u,...,S_{|S^u|-1}^u)$转换成$(S_2^u,S_3^u,...,S_{|S^u|}^u)$,Embedding layer,  self-attention blocks和prediction layer.

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205061519524.PNG" style="zoom:67%;" />

1.首先介绍embedding layer, 首先将序列变成固定长度的序列$s = (s_1, s_2, ...,s_n)$ , 如果超出长度就截断最后的n个,如果小于n就用padding vector在左边补充. 并且创建一个物品embedding矩阵$M\in R^{|I|\times d}$ 输入的input矩阵为$E\in R^{n\times d}$行是序列数n, 列为embedding大小. 由于Self-attention机制没有位置信息,所以要引入position embedding $P \in R^{n\times d}$, 加了position embedding后的input矩阵如下图所示
$$
\hat{E} = \begin{bmatrix} {M_{s_1}+P_1} \\ M_{s_2}+P_2 \\ ...\\ M_{s_n}+P_n\end{bmatrix}
$$
2.接着介绍自注意力块的内容,使用和Transformer一样的dot-product attention, 内容和在Transformer论文的内容
$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d}})V
$$
沿用Transformer的思想, 则输入的$\hat{E}$得到的Self-Attention Layer的输出为, 并且由于序列的先后关系, 所以查询的query和key也需要约定成只query当前和之前的物品.
$$
S = SA(\hat{E})=Attention(\hat{E}W^Q,\hat{E}W^K,\hat{E}W^V)
$$
由于self-attention操作依旧是一个线性操作, 为了让隐维度之间能交互,所以额外引入两层的point-wise feed-forward network对注意力层的输出进行转换, 公式如下所示
$$
F_i=FFN(S_i)=ReLU(S_iW^{(1)}+b^{(1)})W^{(2)}+b^{(2)}
$$
需要整合多个注意力块之间的关系,通过另一个基于F的自我注意块来学习更复杂的项目转换可能是有用的.所以就有如下的公式衔接注意力块
$$
S^{(b)}=SA(F^{(b-1)}) \\
F_i^(b) = FFN(S^{(b)}_i)
$$
当网络更深时，有几个问题:   

1）模型容量的增加导致过拟合  

2）训练过程变得不稳定(由于梯度消失等原因)  

3）具有更多参数的模型往往需要更多的训练时间   

引入了Dropout和残差思想以及LayerNorm的思想. 使用Dropout可以看成是一种集合学习的方式, 考虑了大量共享参数的模型的结果, 残差是为了让low-layer的特征能传播到high-layer, 也就是最初的输入能一直存在到网络的最后.  使用LayerNorm使得用一个sample在一个batch中的归一化是独立的
$$
g(x)=x+Dropout(g(LayerNorm(x))) \\
LayerNorm(x) = \alpha \odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$
3.然后介绍prediction layer的事情, 将之前b层的self-attention block的结果计算相关性.公式如下, $r_{i,t}$代表item$i$ 在给定的$t$个物品的前提下$(s_1,s_2,..,s_t)$的相关性, 而$N$则为另一个物品embedding矩阵, 这个和之前的$M$不同
$$
r_{i,t}=F^{(b)}_tN_i^T
$$
为了降低计算复杂度,将$M$替换$N$, 这种使用同质的物品矩阵在FPMC中会有问题, 因为内积不能表示物品之间的不对称性, 比如买了i后更倾向于买j, 反之不成立. 但是在本文模型中使用同质矩阵不存在这样的问题, 因为最后加了两层的feed-forward network实现非线性转换,可以轻松实现$FFN(M_i)M_j^T \neq FFN(M_j)M_i^T$
$$
r_{i,t}=F^{(b)}_tM_i^T
$$


对于是否需要额外添加用户矩阵(显式或者隐式)的问题,作者实验中发现没有区别, 网络能学到. 作者在后面又说明了训练的一些设置, 以及空间复杂度比起FPMC小, 时间复杂度更少, 并且说明了在处理超长序列上效果并不好. 希望能使用受限的self-attention或者分割长序列.  

最后作者讨论一下SASRec模型和已存在的模型之间的关系, 比如不用self-attention就退化成FMC, 在FMC基础上加上额外的用户矩阵就成了FPMC, 如果只是用一层self-attention并且统一注意力, 并且不用同质的物品矩阵不用position embedding 就变成了FISM.

比起MCs需要提前设定好之前的L个物品数量, SASRec可以自己找到, 并且可以扩展到上百序列长度上.比起RNN, 不仅RNN计算效率低下, 而且它的最大路径长度(从输入节点到相关的输出节点, 见备注2)为O(n), 而本文为O(1),这对学习长路径依赖有益.

**实验结果**:  

复现回来补  

**个人总结**：  

本文算是自注意力推荐的一篇很经典的文章了, 讲解了Transformer中的思想如何运用到推荐系统中, 做了哪些修改的事情, 并且和市面上的模型进行退化比较.

**备注**  

序列推荐(Sequence Recommendation)和时间推荐(Temporal Recommendation)的区别是, 序列推荐只对一系列的行为建模, 而不是基于真正意义上的时间.  

[Transformer/CNN/RNN的对比（时间复杂度，序列操作数，最大路径长度）](https://zhuanlan.zhihu.com/p/264749298)