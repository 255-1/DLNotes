+ ***[BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690)***   

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205071547269.PNG" style="zoom:55%;" />

**年份**：2019  
**引用次数**：366  
**应用领域**：序列推荐  
**方法及优缺点**：

过去的序列推荐的办法是从左到右的推荐, 这种单向推荐作者认为是次优解.因为

1. 单向架构限制了隐藏表示在用户行为序列中的能力
2. 严格的序列可能并不实用

基于这两种限制, 作者提出了BERT4Rec,为了防止信息泄露并且高效地训练双向模型,作者在序列化建模时采用了Cloze目标,通过联合调节左右上下文来预测序列中的随机屏蔽项.通过这种方式,学习了一个双向表示模型,让用户历史行为中的每一项融合来自左侧和右侧的信息

**结论**：

优势同上,  展望了希望能学习到更多的物品特征(最近看的几篇都这么说), 还有一个方向说的是额外的用户建模(类似HRNN?)解决用户有多个session的情况.

**动机**:  

基于[RNN](./Session-Based_Recommendations_with_recurrent_neural_networks.md)的或者是[SASRec](./Self-Attentive_Sequential_Recommendation.md)都假定了session中顺序是严格有序的, 但是事实上可能由于许多无法观察的额外的因素, 并不是如此, 所以需要使用综合上下文的context来推荐. 但是这种训练方式可能会带来信息泄露, 所以引入了Cloze task作为单向推荐的目标, 具体来说就是挖去序列中的一些物品, 然后预测它的id号. 选用Cloze task的另一个原因是因为它可以产生更多的样本去训练.但是有一个问题是,这个目标和普通的序列推荐不一致, 所以要在每个序列的最后都挖掉.

**相关工作和理论**：  

作者介绍顺序是Transformer层, Embedding层, 输出层, 模型学习问题.  

首先介绍Transformer层的内容, 对$t$个输入会通过$t$个Transfomer层得到$t$个特征, 合并在一起成为$H^l\in \mathbb{R}^{t\times d}$, 每个Trm层都有两个组件, 多头注意力和Position-wise Feed-Forward Network(PFFN), Trm层如下图所示

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205071652674.PNG" style="zoom:80%;" />

多头注意力先通过不同的线性变换将$H^l$投射到不同的子空间, 然后将不同头的注意力分数合并, 公式如下所示,和Transformer差不多.
$$
MH(H^l)=[head_1;head_2;..,head_h]W^O \\
head_i=Attention(H^lW^Q_i,H^lW^K_i,H^lW^V_i)\\
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d/h}})V
$$
出于和[SASRec](./Self-Attentive_Sequential_Recommendation.md)中差不多的原因, 引入了PFFN, 唯一的区别是这里用的是GELU以及把多头整合了一下, 公式如下:
$$
PFFN(H^l)=[FFN(h^l_1)T;...;FFN(_t^l)^T]^T\\
FFN(x) = GELU(xW^{(1)}+b^{(1)})W^{(2)}+b^{(2)}\\
GELU(x) = x\Phi(x)
$$
也是出于和[SASRec](./Self-Attentive_Sequential_Recommendation.md)一样的原因, 加了SASRec+Transformer版本的残差,Dropout和LayerNorm, 和SASRec区别是换了换顺序
$$
H^l=Trm(H^{l-1})\\
Trm(H^{l-1})=LN(A^{l-1}+Dropout(PFFN(A^{l-1})))\\
A^{l-1} = LN(H^{l-1}+Dropout(MH(H^{l-1})))
$$
接着介绍了embedding层, 和Transformer一样, 没有位置信息, 所以需要位置信息, 这里的位置信息并不是提前定好的, 也是通过学习得到的.

然后介绍了输出层的, 公式如下, 其中$E$为学习到的item embedding矩阵, 并非额外的,具体原因见[SASRec](./Self-Attentive_Sequential_Recommendation.md)中同质物品矩阵.
$$
P(v)=softmax(GELU(h^L_tW^P+b^P)E^T+b^O)
$$
在模型学习中, 使用Cloze project对序列$t$生成t-1个序列([v1,v2], [v1,v2,v3]),可以有更多的数据用于训练, 如下图所示, 根据概率$\rho$随机mask,

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205071734271.PNG)

那么输出层在被mask的位置生成的item就可以使用如下的loss function,计算损失, $S'_u$为被mask的输入, $P(\cdot)$就是Eq4的公式.
$$
\mathcal{L}=\frac{1}{|S^m_u|}\sum_{v_m\in S^m_u}-logP(v_m=v_m^*|S'_u)
$$
因为Cloze task的目标是预测当前被屏蔽的商品,而顺序推荐的目标是预测未来. 为了解决这个问题, 作者在用户行为序列的末尾附加一个特殊的mask，然后根据这个mask的隐藏物品表示来预测下一个项目.并且作者还生成了只mask最后一个item的数据用于训练.

最后作者对比了BERT4Rec和SASRec,CBOW&SG和BERT的区别. SASRec只是单向的. CBOW只是学到了单个的word的特征而不是一个序列的特征. BERT只是预训练了句子特征,但是本文是端到端的并且是适用于推荐系统的, 不可以只训练特征, 并且由于BERT4Rec将用户的所有历史行为当成一个序列, 所以移除了BERT的next sentence loss和segment embedding.

**实验结果**:  

复现回来补

**个人总结**：  

个人感觉把Transformer+BERT+SASRec装在一起形成了这篇论文, 大致了解了Transformer的架构和SASRec不难理解这篇文章. 也算是NLP和推荐系统关系很紧密的一种表现

**备注**  