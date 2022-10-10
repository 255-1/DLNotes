# 论文笔记

## 对比学习基础

+ ##  [Unsupervised Feature Learning via non-Parametric Instance Discrimination(InstDisc)](./Paper/Unsupervised_Feature_Learning_via_Non-Parametric_Instance_Discrimination.md)

  > 无需数据标签的无监督学习,通过instance-level级别的学习,直接学习图片的特诊,在下游任务中通过fine-tuning也可以有很好的结果,并且此方法对计算复杂度和存储要求较低  

+ ##  [Unsupervised Embedding Learning via Invariant and Spreading Instance Feature(InvaSpread)](./Paper/Unsupervised_Embedding_Learning_via_Invariant_and_Spreading_Instance_Feature.md)

  >传统的无监督学习学习到的所谓的"intermediate"特征可能无法保持视觉上的相似性,基于相似性的任务性能会急剧下降.本文提出的Siamese架构旨在学习到数据增强的Invariant和个体的spread-out特征.并且直接通过优化最顶层的softmax函数达到快速学习并且高准确率的目的.在训练测试集没见过的任务上也表现得十分良好.  

+ ##  [Representation Learning with Contrastive Predictive Coding(CPC)](./Paper/Representation_Learning_with_Constrastive_Predictive_Coding.md)

  >CPC作为特征提取可以用于任何领域,这些数据可以以有序的顺序表示: 文本、语音、视频,甚至图像(图像可以看作是一系列像素或补丁). CPC通过编码信息来学习表示,这些信息在多个时间步骤之间共享,而丢弃本地信息.这些特性通常被称为“慢特性”: 这些特性不会随着时间的推移而变化得太快.主要优点简单,计算需求小并且结果很encouraging, 而且这个用法可以用在许多形式.  

+ ##  [Contrastive Multiview Coding(CMC)](./Paper/CMC.md)
  
  >通过最大化同一物品的不同视角下的mutual information学习到的特征更加好, 视角越多效果越好, 作者推广了之前的只有两个视角的方法到更多视角的情况([CPC](./Paper/Representation_Learning_with_Constrastive_Predictive_Coding.md), DeepInfoMax, [InstDisc](./Paper/Unsupervised_Feature_Learning_via_Non-Parametric_Instance_Discrimination.md)).   
  
+ ##  [Learning deep representations by mutual information estimation and maximization(DeepInfoMax)](./Paper/DeepInfoMax.md)

  > 本文通过最大化输入和输出之间的互信息MI(mutual information)来无监督学习representation, 并且作者发现结构很重要, 整合输入中的局部信息(local MI)可以显著提高下游任务的适用性. DIM同时考虑了global MI, local MI, 和使用AAE(Adversarial autoencoders)思想限制先验概率,来约束学习到得representation, 让它得分布更容易处理.

+ ##  [Mutual Information Neural Estimation(MINE)](./Paper/Mutual_Information_Neural_Estimation.md)

  > 作者通过找到了一种办法可以通过神经网络计算高维连续的变量之间的互信息, 这种新的办法在维度和样本量上都是线性可伸缩的, 可以通过back-prop进行训练并且有强一致性.  

+ ##  [Momentum Contrast for Unsupervised Visual Representation Learning(Moco)](./Paper/Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning.md)

  >Moco将对比学习特征看成一个字典查找问题, 并且总结了前人的问题, 通过使用队列+动量编码器实现了一个又大又一致的的字典办法, 前人的成果只能有其中一种办法.这些特征在下游任务上效果很好.
  
+ ## [Prototypical Contrastive Learning Of  Unsupervised Representation(PCL)](./Paper/PCL.md)

  > PCL在个体判别的基础上引入聚类算法, 这样不仅可以学习到low-level的特征区别还可以学习到隐语义结构信息. 对于聚类算法使用EM算法来极大似然估计来优化网络参数$\theta$ 

对比学习有价值的内容  

[李沐Moco](https://www.bilibili.com/video/BV1C3411s7t9/?spm_id_from=333.788)  

[对比学习综述](https://www.bilibili.com/video/BV19S4y1M7hm/?spm_id_from=333.788)  

[对比学习综述博客](https://ankeshanand.com/blog/2020/01/26/contrative-self-supervised-learning.html)  

[[译] Noise Contrastive Estimation](https://zhuanlan.zhihu.com/p/76568362/)    

[Noise Contrastive Estimation 前世今生——从 NCE 到 InfoNCE](https://zhuanlan.zhihu.com/p/334772391)  

[理解Contrastive Predictive Coding和NCE Loss](https://zhuanlan.zhihu.com/p/129076690)  

[超直观无公式图解contrastive predictive coding从脸盲说起](https://zhuanlan.zhihu.com/p/177883526)  

[浅析Contrastive Predictive Coding](https://zhuanlan.zhihu.com/p/137076811)    

[对 CPC (对比预测编码) 的理解](https://zhuanlan.zhihu.com/p/317711322)

[真正的无监督学习之一——Contrastive Predictive Coding](https://zhuanlan.zhihu.com/p/75517749)    

[NCE的配分函数解释](https://kexue.fm/archives/5617/comment-page-1)

[Representation Learning with Contrastive Predictive Coding](https://zhuanlan.zhihu.com/p/461505149)  

[The Illustrated Word2vec(感觉是对比学习的一些前置任务)](https://jalammar.github.io/illustrated-word2vec/)  

[对比学习二 | Unsupervised Embedding Learning via Invariant and Spreading Instance Feature](https://zhuanlan.zhihu.com/p/459345219)  

[DIM作者youtube讲解](https://www.youtube.com/watch?v=o1HIkn8LEsw&t=256s)  

[DIM理解](https://zhuanlan.zhihu.com/p/277660074)  

[MINE](https://zhuanlan.zhihu.com/p/113455332)   

[MINE——基于神经网络的互信息估计器](https://zhuanlan.zhihu.com/p/191155238)  

[Mutual Information Neural Estimator(MINE)：通过样本有效估计高维连续数据互信息](https://zhuanlan.zhihu.com/p/412538959)  

[Explanation of Mutual Information Neural Estimation](https://ruihongqiu.github.io/posts/2020/07/mine/)  

[对比学习（Contrastive Learning）:研究进展精要](https://zhuanlan.zhihu.com/p/367290573)

## 序列推荐

+ ##  [Session-Based Recommendations with recurrent neural networks(GRU4Rec)](./Paper/Session-Based_Recommendations_with_recurrent_neural_networks.md)

  > 这篇文章是RNN首次使用在RS领域的文章,作者认为现实生活的RS通常只能是 short session-based数据. 这种情况下常用的矩阵分解技术并不准确, 在实际工程里这个问题通常通过推荐相似物品来解决. 作者通过对整个session建模可以得到更好的准确度, 并通过引入一些修改来让RNN适应任务.

+ ## [Improved Recurrent Neural Networks for Session-based Recommendations](./Paper/ImprovedGRU4Rec.md)

  > 使用了NLP和CV领域中的方法改善[GRU4Rec](./Session-Based_Recommendations_with_recurrent_neural_networks.md),使用数据增强以及一种考虑了输入数据变换的方法, 并且作者借用了CV中的teacher-student用于预训练和微调, 最后作者并不是预测one-hot编码的物品, 而是直接预测下一个物品的embedding. 得到了很好的效果

+ ## [Recurrent Neural Networks with Top-k Gains for Session-based Recommendations](./Paper/Recurrent_Neural_Networks_with_Top-k_Gains_for_Session-based_Recommendations.md)

  > 本文在之前的GRU4Rec的基础上改进了loss function使得梯度消失的问题得到缓解, 并且这种新办法并不会增加太多的计算代价  

+ ## [Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding(Caser)](./Paper/Caser.md)

  > Caser使用了CNN应用于NLP的思想提取序列的特征,既学习到序列模式的特征又捕获用户偏好, 具体来说序列模式能捕获到point-level和union-level两种, 用户偏好能捕获到skip behavior. 简单来说就是再对用户点击过的物品建模,得到embedding的基础上, 再加一个用户向量作为一般偏好一起训练得到结果.

+ ## [Neural Attentive Session-based Recommendation(NARM)](./Paper/Neural_Attentive_Session-based_Recommendation.md)

  > 前人在序列推荐中只考虑了整个序列特征, 本文在之前的基础上使用注意力机制额外整合了用户的意图(NARM).

+ ## [Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks(HRNN)](./Paper/Personalizing_Session-based_Recommendations_with_Hierarchical_Recurrent_Neural_Networks.md)

  > 在session-based推荐系统中用户的行为可能很难获取, 但是也可能已经拥有一些可用的数据. 作者提出了一个无缝的个性化RNN模型可以处理session之间的信息转换, 并且设计了一个分层的RNN模型结构, 该模型可以在用户的session之间传递和演化隐藏状态.

+ ## [Self-Attentive Sequential Recommendation(SASRec)](./Paper/Self-Attentive_Sequential_Recommendation.md)

  > 为了获取用户行为的上下文context信息, 有两大类的办法被提出, 第一种是马尔可夫链的方式(MCs), 在处理稀疏数据上效果比较好, 第二种是基于RNN的, 在大量数据上有更好的效果, 本文的目标SASRec是平衡两种目标, 使用自注意力机制让我们既能像RNN一样捕获到长期的语义, 也能像MCs一样基于过去的部分活动. 在SASRec中想要找到和用户历史行为相关的物品, 用于预测下一次点击的物品.

+ ## [STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation](./Paper/STAMP.md)

  > 现有的LSTM方法没有考虑用户当前的行为对下一个行为的影响. 作者认为long-term模型可能不足以对长期会话进行建模，这些会话通常包含由非故意点击引起的用户兴趣漂移. 本文的STAMP模型既可以获取用户长session的普遍兴趣, 也考虑了用户最后一次点击所代表的用户兴趣.STAMP强调最后一次点击所反映的用户兴趣.短期兴趣可以在STAMP中得到加强，以便在兴趣变化的情况下准确地捕获用户的当前兴趣，特别是在一个long session中.

+ ## [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](./Paper/BERT4Rec.md)

  > 过去的序列推荐的办法是从左到右的推荐, 这种单向推荐作者认为是次优解.因为
  >
  > 1. 单向架构限制了隐藏表示在用户行为序列中的能力
  > 2. 严格的序列可能并不实用
  >
  > 基于这两种限制, 作者提出了BERT4Rec,为了防止信息泄露并且高效地训练双向模型,作者在序列化建模时采用了Cloze目标,通过联合调节左右上下文来预测序列中的随机屏蔽项.通过这种方式,学习了一个双向表示模型,让用户历史行为中的每一项融合来自左侧和右侧的信息

+ ## [Feature-level Deeper Self-Attention Network for Sequential Recommendation(FDSA)](./Paper/FDSA.md)

  > 现有的模型只考虑了序列中物品的转换模式, 忽视了物品的特征(种类, 品牌)之间的转换模式. 所以提出了本文的FDSA模型, 首次将不同的物品特征整合进特征序列中, 将推荐过程分成item-level和feature-level两个方向, 并且各自使用自注意力块进行学习.  最后整合到一起经过两层的全连接层进行物品推荐.

序列推荐有价值的内容  
[序列推荐论文推荐](https://zhuanlan.zhihu.com/p/389044011)

[GRU讲解](https://www.youtube.com/watch?v=T8mGfIy9dWM&t=1952s)

[self-attention入门](https://www.youtube.com/watch?v=hYdO9CscNes)    

[超详细图解Self-Attention](https://zhuanlan.zhihu.com/p/410776234)  

[This post is all you need](https://github.com/255-1/PaperRecords/blob/main/PaperRecords.md)  

[李沐Transformer](https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.788)  

[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)  

[推荐中的序列化建模：Session-based neural recommendation](https://zhuanlan.zhihu.com/p/30720579)

[Neural-Attentive-Session-Based-Recommendation](https://www.jianshu.com/p/7c0cc424d06c)



## GNN基础

- ## [Convolutional	 Neural Networks on Graphs with Fast Localized Spectral Filtering(GCN)](./Paper/GCN.md)

  > 本模型旨在将CNN推广到图这种数据结构中, 在spectral graph理论的基础上使用CNN, 这种办法有和CNN一样的计算复杂度和学习复杂性, 并且这种办法可以适用于任何一种图结构中.

- ## [SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](./Paper/Semi_GCN.md)

  > 提出了一种在图结构上进行半监督学习的方法, 它使用切比雪夫的一阶近似估算, 模型复杂度和边数成线性, 并学习了编码局部图形结构和节点特征的隐藏层表示.

- ## [GRAPH ATTENTION NETWORKS(GAT)](./Paper/GAT.md)

  > 在图结构数据上使用self-attention机制, 能自己学习当前节点和周围节点的权重. 并且本文解决了spectral graph的几个关键问题, 使得GAT模型可以应用于inductive和transductive问题.

- ## [Inductive Representation Learning on Large Graphs(GraphSAGE)](./Paper/GraphSAGE.md)

  > [GCN](./Paper/GCN.md)这种办法处理不了动态图, 训练时就需要包含全部的节点, 对于unseen节点需要重新训练, 所以提出了本文的GraphSAGE, 一种inductive learning的方法, 本模型可以通过sampling和aggregating节点周围的邻居节点学习未见过的节点的feature.



gnn值得一读的内容

[李宏毅助教](https://www.bilibili.com/video/BV1G54y1971S?spm_id_from=333.337.search-card.all.click)  

[李沐GNN](https://www.bilibili.com/video/BV1iT4y1d7zP?spm_id_from=333.337.search-card.all.click)

[distill博客](https://distill.pub/2021/gnn-intro/)

[如何理解 Graph Convolutional Network -superbrohter-日智](https://www.zhihu.com/question/54504471/answer/332657604)

[GCN (Graph Convolutional Network) 图卷积网络解析](https://blog.csdn.net/weixin_36474809/article/details/89316439)

[如何理解拉普拉斯变换？](https://www.matongxue.com/madocs/723/)

[向往的GAT(图注意力网络的原理、实现及计算复杂度)](https://zhuanlan.zhihu.com/p/81350196)

[GraphSAGE：我寻思GCN也没我牛逼](https://zhuanlan.zhihu.com/p/74242097)

以下GNN的内容为进阶

[图卷积神经网络(Graph Convolutional Network, GCN)](https://blog.csdn.net/a358463121/article/details/88921154)

[谱图理论(spectral graph theory)](https://blog.csdn.net/a358463121/article/details/100166818)

[Spectral graph theory](https://en.wikipedia.org/wiki/Spectral_graph_theory)



## GNN+序列推荐

- ## [GATED GRAPH SEQUENCE NEURAL NETWORKS(GGNN)](./Paper/GGNN.md)

  > 将GRU拓宽到GNN中间, 让GNN能够处理序列数据得到对应的feature.

- ## [Session-based Recommendation with Graph Neural Networks(SR-GNN)](./Paper/SRGNN.md)

  > 之前的session-based推荐方法会建模一个序列并且也有考虑user representation, 但是这个user representation不够准确并且忽视了物品的复杂过度情况, 为了获取更加精准的嵌入我们提出了本文的SR-GNN模型. 将session的序列编码成图结构的形式, 变成序列图, 这样可以得到物品的复杂转换. 每个session都会用注意力机制编码成全局和当前兴趣的整合.

- ## [Graph Contextualized Self-Attention Network for Session-based Recommendation(GC-SAN)](./Paper/GC-SAN.md)

  > 使用Transformer代替了[SRGNN](./Paper/SRGNN.md)中的生成session部分, 其他一毛一样.按照作者的话就是Transformer可以学到更好的session global feature

- ## [Rethinking the Item Order in Session-based Recommendation with Graph Neural Networks(FGNN)](./Paper/FGNN.md)

  > 和[SRGNN](./SRGNN.md)一样都是将序列转换为序列图, 但是FGNN是将一个序列转换成一个图, 不同序列之间的图独立, 然后使用Readout操作得到序列图的特征, 将序列推荐问题转成了图分类问题.  除此以外作者提出了weighted attention graph layer来更新节点embedding.

- ## [Personalized Graph Neural Networks with Attention Mechanism for Session-Aware Recommendation(A-PGNN)](./Paper/A-PGNN.md)

  > 对于session-aware推荐本模型提出了两个主要组件. 第一, PGNN对用户的序列图额外考虑了用户特征. 第二使用Trm融合了历史session对当前session的影响.

## GNN+CF

- ## [Neural Graph Collaborative Filtering(NGCF)](./Paper/NGCF.md)

  > 学习用户/物品的特征是非常重要的一件事情, 现在的方法都是通过学习物品ID和属性的办法获得, 但是没有考虑用户和物品交互之间的关系, 作者将这种关系叫做collaborative signal. 本文通过学习物品-用户交互的二部图. 主要贡献就是把GNN和CF合在一起.

- ## [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](./Paper/LightGCN.md)

  > 作者发现GCN中最常用的设计, 特征变换和激活函数其实没什么用,所以作者提出了LightGCN只包含最重要的组件邻居消息聚合, 在[NGCF](./Paper/NGCF.md)的基础上简化完成

## GNN+推荐

- ## [Graph Convolutional Neural Networks for Web-Scale Recommender Systems(PinSage)](./Paper/PinSage.md)

  >  工业上对大规模数据集实现GCN的方法30亿节点,180亿条边,75亿个样本,本文结合了随机游走和图卷积生成了节点的embedding, 并且作者使用了hard negative sample提升鲁棒性并取得到不错的效果, 除此以外作者通过importance pooling和curriculum training 提高了embedding的效果.

[负样本为王：评Facebook的向量化召回算法](https://zhuanlan.zhihu.com/p/165064102)

[PinSAGE 召回模型及源码分析(1): PinSAGE 简介](https://zhuanlan.zhihu.com/p/275942839)

[PinSage：GCN在商业推荐系统首次成功应用](https://zhuanlan.zhihu.com/p/63214411)



## 对比学习 + CF

- ## [Bootstrapping User and Item Representations for One-Class Collaborative Filtering(BUIR)](./Paper/BUIR.md)

  > BPR的方法为了判别正样本和负样本, 前人工作更多的依赖于负样本采样, 但是这种情况下可能会让"未被观察的正样本"定义为负样本. 本文的BUIR提出了一种不需要负样本, 不仅让正样本之间的相关性更强, 也能防止模型塌陷.
  >
  > BUIR有两个encoder, 第一个online encoder用来预测第二个encoder的输出, 第二个target encoder通过慢慢近似第一个encoder提供一个稳定的目标.BUIR通过直接最小化item和user的交叉预测误差来学习特征.  除此以外BUIR使用数据增强input来缓解数据稀疏问题.

- ## [Self-supervised Graph Learning for Recommendation(SGL)](./Paper/SGL.md)

  > 作者认为[LightGCN](./LightGCN.md)和[PinSage](./PinSage.md)有两个限制, 高度数节点在特征学习上作用更强. 损害了低度数节点的特征学习, 其次, 特征容易受到噪声影响, 由于agg的方案会扩大观察到的边的影响. 所以为了加强鲁棒性和准确度增加了辅助的自监督任务, 设计了三种生成视角, **node dropout, edge dropout, random walk**, 除此以外作者还发现使用硬负样本会有其他作用, 不仅提升了模型表现还加速了训练过程.  
  
- ## [Are Graph Augmentations Necessary? Simple GraphContrastive Learning for Recommendation(SimGCL)](./Paper/SimGCL.md)
  
  > 作者想要知道为什么CL方法能得到更好表现效果, 得到在基于CL的推荐模型中，CL通过学习更均匀分布的用户/物品表示来操作，这可以隐式地减轻受欢迎程度偏差, 同时作者得到了图的增强并不是必须的, 反而是添加一些均匀的噪声到embedding里面会得到更好的效果. 最后作者得到在基于CL的模型中, CL loss是核心, 而图增强只是一些次要角色, 优化CL loss有助于在推荐场景下去偏见.

## 对比学习+序列

- ## [Contrastive Learning for Sequential Recommendation(CL4Rec)](./Paper/CL4SRec.md)

  > 普通的序列推荐由于数据的稀疏性导致很难学习到高质量的用户特征. 所以引入了对比学习框架去捕获自监督信号. 本论文使用了三种数据增强的方法(crop/mask/reorder)去生成自监督信号.

- ## [S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization](./Paper/S3Rec.md)

  > 普通的序列推荐模型容易受到数据稀疏问题. 所以提出了本文的S3Rec.该方法的主要思想是利用内在的数据相关性来获得自我监督信号，并通过预先训练的方法来增强数据表示，以改进序列推荐.  使用了自监督目标和MIM来学习不同的属性,物品, 子序列和序列这些的关联. MIM可以提供一个统一的方法去表现数据的关联性. 这是第一个序列推荐的pre-train的序列推荐模型, 自监督目标为**item-attribute,sequence-item, sequence-attribute和sequence-subsequence**

- ## [Disentangled Self-Supervision in Sequential Recommenders(DSSRec)](./Paper/Disentangled_Self-Supervision_in_Sequential_Recommenderss.md)

  > 本文的seq2seq在传统的seq2item策略上进行加强, 能发现更加long-term future, 而不是停留在下一个点击预测.

- ## [Intent Contrastive Learning for Sequential Recommendation(ICL)](./Paper/ICL.md)

  > 将[PCL](./Paper/PCL.md)中的聚类用于推荐系统当成使用户的intention用于序列推荐, 通过最大化intention和序列这一对正样本的相似度来完成对比学习. 

- ## [Enhancing Sequential Recommendation with Graph Contrastive Learning(GCL4SR)](./Paper/GCL4SR.md)

  > 类似SRGNN的做法, 对全局的序列有向图设计了一个新的点击频率的邻接矩阵的做法, 相当于对SRGNN做了个对比学习, 除此以外因为是序列推荐引入了用户信息, 使用一个User Specific Gating融合用户的信息, 这里用了MMD loss来融合全局的信息和当前序列的信息
  
- ## [Sequential Recommendation with Multiple Contrast Signals(ContraRec)](./Paper/ContraRec.md)

  > 使用了两种对比context-target和context-context, 前一种希望序列和目标物品相似, 后一种希望相同物品的前置序列要尽量相似. 除此以外, context-target的方法可以看成是传统BPR的一种推广作者这里提出了BPR+新的loss. 	

- ## [Self-Supervised Graph Co-Training for Session-based Recommendation](./Paper/COTREC.md)

## 综述

- ## [Toward the next generation of recommender systems: a survey of the state-of-the-art and possible extensions](./Paper/Toward_the_next_generation_of_recommender_systems_a_survey_of_the_state-of-the-art_and_possible_extensions.md)

- ## [Graph Neural Networks in Recommender Systems: A Survey](./Paper/GNNRec_Survey.md)

- ## [Self-Supervised Learning for Recommender System: A Survey](./Paper/SSR_Survy.md)



其他值得一读的文章  

[研究推荐系统要对NLP很了解吗？](https://www.zhihu.com/question/317441966)  

[少数派报告：谈推荐场景下的对比学习](https://zhuanlan.zhihu.com/p/435903339)

