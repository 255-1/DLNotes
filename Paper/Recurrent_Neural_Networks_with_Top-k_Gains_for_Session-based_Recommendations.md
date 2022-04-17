+ ***[Recurrent Neural Networks with Top-k Gains for Session-based Recommendations](https://arxiv.org/abs/1706.03847)***   

**年份**：2017    

**引用次数**：347    

**应用领域**：session-based recommendation system  

**方法及优缺点**：  

本文在之前的GRU4Rec的基础上改进了loss function使得梯度消失的问题得到缓解, 并且这种新办法并不会增加太多的计算代价  

**结论**：
作者认为新提出的这个loss function能更加普适,并且与相应的采样策略一起还为不同的推荐设置(矩阵分解, AE)提供了top-k. 以及作者认为推荐系统和NLP是类似的, 将这个新的loss function与其他的改进办法(embedding)可以得到比之前更好的解决方案.    

**动机**:  

由于深度学习方法需要传播梯度, 源自损失函数的这些梯度的质量会影响优化的质量和模型参数,所以损失函数很重要. 此外, 推荐系统通常需要很大的output space,因为项目多, 在设计损失函数的时候也需要考虑这个问题.    

**相关工作和理论**： 

对于GRU4Rec有很大的副作用, 由于要把一个session分解为多个sub-session会增加大量的训练时间.并且GRU4Rec中使用的BPR loss function根据动机中的描述也需要进行更加适合推荐系统的调整.    

所以本文的目标有:    

1\) 解决输出的物品数量极大的问题    

2\) 提出新的loss function  

回顾GRU4Rec中的抽样机制, 输入的是session中的每一个event的one-hot编码, 而输出是根据流行度抽样出的物品,这些物品是整体的一个子集, 基于这种办法的计算复杂度为$O(N_E(H^2+HN_O))$ $N_E$为event数量, $H$为hidden units, $N_O$为输出物品数量.此外GRU4Rec使用mini-batch并行计算不同样本, 下图展示了是如何操作的,类似交叉熵.  

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204161752731.PNG)  

对于这种抽样机制, 只有当目标项目的分数没有超过负样本的分数时才学习,因此包含高分的负样本很重要,在很多情况下, 热门项目通常得分较高, 这使得GRU4Rec的流行度采样很好. 但是这种抽样方式的问题有, 当target的学习的分数超过了流行物品后学习速度慢, 对于长尾高分的排名不太准确. 另一方面低分负样本占据大多数, 均匀抽样的学习速度很慢.   

综上mini-batch的三个限制有:1) mini-batch通常很小, 如果项目数量大, 小样本不太能包含所有高分负样本. 2) mini-batch对训练速度有直接影响. 3) 基于流行度的抽样办法可能并非对所有数据集都是最优的.  

所以作者增加了样本, 每个example都将这些增加的样本作为负样本, 额外增加的样本可以来自$sup p_i^\alpha$其中$\alpha=0$为均匀采样, 1为流行度采样. 这种增加采样的办法必会增加时间复杂度. 然而由于并行化处理所以训练时间不会增加,但是GPU上的分布采样没有很好的支持, 所以作者这里实现了一个缓存,进行预采样并且存储大量的负样本来提高训练速度.由此完成目标1)  

接着作者开始说明loss function. GRU4Rec的loss function很弱, 随着输出的数量增加会退化, 所以作者基于两种办法来稳定交叉熵损失函数的不稳定性.  

GRU4Rec有引入softmax和交叉熵结合的loss function.如下图所示,但是这种办法不稳定,当有一个物品的分数$r_k >> r_i$ 时log内趋于0. loss function未定义.可以通过两种办法缓解这个问题. a)计算$-log(s_i+\epsilon)$但是引入了噪声. b) 直接计算$-r_i+log\sum_{j=1}^Ne^{r_j}$ .这两种办法都可以稳定loss function并且结果没有显著的差别.  

$L_{xe}=-logs_i=-log\frac{e^{r_i}}{\sum_{j=1}^Ne^{r_j}}$    

接着讨论GRU4Rec引入的TOP1&BPR loss function的梯度消失问题.这两loss的梯度公式如下图所示, 对于不相关的物品$r_j << r_i$ .下式的$\sigma(r_j-r_i)$和$(1-\sigma(r_i-r_j))$都趋于0, 让整个loss不再更新, 所以低分负样本无法提供足够的梯度, 而且所有梯度都需要除以负样本数量$N_s$ 而且$N_s$也不能简单的去除, 因为取消了归一化整体学习也会不稳定.   此外还可以注意到TOP1面对相关的物品$r_j>> r_i$也会梯度消失  

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204171041178.PNG)  



基于上述梯度消失问题作者提出了ranking-max loss function解决办法.其思想是将目标分数与最相关的样本分数进行比较, 通用公式如下所示, 由于max无法微分所以使用softmax代替.由此可以推导TOP1-max和BPR-max的loss function.  

$L_{pairwise-max}(r_i, \{r_j\}^{N_s}_{j=1})=L_pairwise(r_i, maxr_j)$  

TOP1-max公式如下, 在原来的top1基础上增加了sofrmax score $s_j$ 权重,分数接近最大值的例子将获得更多的权重。这解决了用更多样本消失梯度的问题，因为不相关的样本将被忽略.  

$L_{top1-max}=\sum^{N_s}_{j=1}s_j(\sigma(r_j- r_i)+\sigma(r_j^2))$    

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204171128165.PNG)  

BPR-max公式如下,和上面一样都是使用了$s_j$的加权平均  

$L_{bpr-max}=-log\sum^{N_s}_{j=1}s_j\sigma(r_j- r_i)$  

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204171240628.PNG)  

TOP1-max和BPR-max都和softmax score成比例并且只有接近最大值的的物品才会被更新,下图展示了梯度和rank排名的关系, 越低的rank代表负样本越少. 当rank小代表相关物品多,左图的BPR更好, 因为max只看了最相关的,忽略了其他相关的, 随着rank变大(中, 右), 相关物品越来越少, BPR梯度快速消失, 大约在rank5消失.    

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204171430514.PNG)  

GRU4Rec提供dropout和$l_2$正则, 作者做了实验发现$l_2$正则影响模型结果, 作者假设一些模型权重——比如用于计算更新和重置门的权重矩阵——不应该被正则化. 惩罚高输出分数会约束模型, 即使没有显式正则化权重.    

作者在这里给BPR-max添加正则项,至于为什么不是TOP1-max,我不太清楚,作者将样本分数设置为独立的、零均值的高斯函数，其方差与softmax分数成反比,公式如下图所示.(这里公式还有待进一步理解).  

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204171539353.PNG)  

最后得到最终的BPR-max的loss function,softmax加权$l_2$正则化对负样本的得分。λ是损耗的正则化超参数  

$L_{bpr-max}=-log\sum^{N_s}_{j=1}s_j\sigma(r_j- r_i)+\lambda\sum^{N_s}_{j=1}s_jr_j^2$  

**实验结果**:  

复现回来补  

**个人总结**： 

这篇论文算是在之前的GRU4Rec的基础上的进一步拓展, 其中提到的RS和NLP有异曲同工之处,我个人还是很认可这个观点的  

**备注**:  