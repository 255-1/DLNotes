+ ***[Disentangled Self-Supervision in Sequential Recommenders](https://dl.acm.org/doi/abs/10.1145/3394486.3403091)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081311887.PNG)

**年份**：2020  

**引用次数**： 45  

**应用领域**：self-supervision, sequential recommender  

**方法及优缺点**：   
本文的seq2seq在传统的seq2item策略上进行加强, 能发现更加long-term future, 而不是停留在下一个点击预测.

**结论**：
本文通过在隐空间中的自监督训练以及分解用户intentions来充分利用来自long-term future的额外监督信号, 这能带来额外的收益. 展望了未来提高长序列上的性能以及降低计算成本的问题.

**动机**:  
seq2item在一些情况下无法推荐出多样化的物品,文中给的例子，交互序列为：“shirt, shirt, shirt, shirt, shirt, trousers”。这个交互序列会导致很多连续子序列的label为shirt，而只有极少数的label为trousers。于是，在用户点击了shirts之后，推荐系统可能会更多的倾向于给用户推荐shirt，而作者希望的是模型可以推荐得更加均衡.其次seq2item的下一次点击与之前点击无关时会十分脆弱.  
由此发现的主要挑战有1)重建序列可能会有大量的冗余信息导致收敛困难,2)所有未来序列可能来自许多的intentions,并不是所有未来的选择都和先前的选择有关联.  
为了解决这些办法作者使用了隐空间的自监督学习而不是原始数据上自监督学习,并将给定用户的前面交互的序列表示希望模型预测出未来交互子序列的表示（主要是给定一个用户的子序列希望其预测另一个行为子序列,以此解决1)的冗余问题  
设计了一个sequence encoder它可以推断和分解给定行为序列所反映的潜在意图. disentangled encoder 输出给定行为序列的多个表示，其中每个表示集中于给定序列的不同子序列,每个子序列代表用户不同的intention, 以此完成2).

**相关工作和理论**：  
主要目标有:  
1)设计seq2seq策略.    
2)设计一个sequence encoder,可以推断和分解给定行为序列所反映的潜在意图  
3)设计一个disentangled encoder,输出给定行为序列的多个表示，其中每个表示集中于给定序列的不同子序列    

变量定义如下图所示  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081312116.PNG)  
一个encoder $\varPhi_\theta(\cdot)$，有一个item embedding table $H \in R^{M \times D}$ . encoder以 $x_u$ 作为输入，输出其表示 $\varPhi_\theta(x_u)$ , 这可以看成是user $u$ 的intentions表示。这里的输出为 $K$ 个 $D$ 维向量，表示用户的intention在 $K$ 个latent category下的表示；然后通过评估 $\varPhi_\theta(x_u)$ 和每个商品的embedding $H_i$ 相似度来进行推荐。  
对于传统的seq2item的loss计算如下图所示,对于所有的user $u$ 和时间点$t$最大化在之前点击的条件下当前时刻$t$的概率.    
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081312242.PNG)  
接着作者引入了seq2seq策略为了完善seq2item,首先这里假设我们已经训练好了encoder(后面讲),作者使用$\varPhi_\theta(\cdot) = \{\varPhi_\theta^{(k)}(\cdot)\}_{k=1}^K$代表用户经过encoder得到的$K$个不同intentions下的$D$维向量,如果在某个intention下缺失就用白噪声代替.则seq2seq的loss如下图所示, 分子的序列翻转是因为encoder的设计让靠近当前时刻$t$的item的权重更加大,分子内积通过$\frac{1}{\sqrt{D}}$做标准化,因为encoder最后为LayerNormalization(LN).  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081312942.PNG)  
接着作者提到了只用可信样本进行训练, 为此只计算$x_{1:t}^{(u)}$和$x_{T_u:t+1}^{(u)}$encoder之后的K个intention都有的情况,并且使用一个超参数$\lambda$作为置信区间, 公式如下图所示.  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081313048.PNG)  
接着作者提出了他们的seq2item,为了让训练并非在实际数据,而是在隐空间上进行计算,降低对于数据边界对齐问题的依赖,seq2item的loss公式如下图所示,使用encoder后的向量和物品的embedding代替了原来的$x$.  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081313603.PNG)  
综上,整合了seq2item的loss和seq2seq的loss的整体loss公式如下图所示, 由此完成了1)的目标:  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081313645.PNG)  

接着作者介绍他们的disentangled sequence encoder是如何实现的,也就是如何把用户点击的序列数据分成不同intentin下的子序列.  
作者发现SASRec(self-attentive sequential recommendation)中的multi-head比起single-head效果没有太大提升.这两个都倾向于推荐用户最后一次点击的物品的相似物品.所以作者在single-head的基础上添加了一层intent-disentangled layer加强表达能力, 用$z_i^{(u)}$为来自single-head SASRec encoder的结果,这个变量可以看成是用户点击当前物品intention的集合体.  
则作者提取的加强版encoder分为三步走, intention聚类, intention权重, intention聚合.  
对于intention聚类,计算点击i得到当前intention集合体情况下intention为K的概率, 通过计算intention到一些intention prototypes的距离得到结果,公式如下图所示, 分子的相似度度量实则为cos相似度,因为比起dot相似度,cos相似度能防止模型塌陷.
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081313539.PNG)   
对于intention权重, 计算了点击i得到当前intention集合体情况下这个集合体对于未来的intention的重要性,公式如下图所示, 其中$\alpha$为序列的position-embedding, 使用这个$\alpha$是因为基于两个假设,最近的点击更加有价值,在向量空间中接近最新意图的向量更加重要, 这两个假设不一定一直正确所以引入了$W,b,b'$调整  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081313362.PNG)  
个人理解intention权重的计算过程如下图所示,  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081314936.PNG)  
最后是intention aggregation,有$p_i$和$p_{k|i}$就可以计算为intention $k$的概率,公式如下图所示,$\beta$为输出k的偏差,使用$(0, \frac{1}{\sqrt{D}})$的正态分布的数据.对于过去的点击数据$x^{(u)}_{1:t}$和未来的点击数据$x^{(u)}_{T_u:t+1}取不同的$\beta$  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081314928.PNG)  

对于disentanglement不需要额外的loss, 一般而言希望引入一个正则项对不同intention的MI做最小化.但是在Eq.3和Eq.6的loss中分子的计算是同一个intention相似度,而分母要计算不同intention下的总和,所以这个loss已经要求模型相同intention下相似度尽量大,不同intention下相似度尽量小,所以不需要额外对MI做最小化.  

**实验结果**:  
复现再回来补

**个人总结**： 
这文章公式看着很唬人,而且没有开源代码,知乎有人说这是大水文,我看得少,不好做评价.  

**备注**:  
