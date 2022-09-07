+ ***[Self-Supervised Learning for Recommender Systems: A Survey](https://arxiv.org/abs/2203.15876)***   

![image-20220906112457556](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906112457556.png)

**年份**：2022

**引用次数**： 6

**应用领域**：  SSL推荐系统综述

**相关工作和理论**：  

Self-Supervised learning (SSL), 作为一个学习模板可以降低对人工数据的依赖, 并且可以在大量的非标签数据上进行训练.  早期的SSR可以追溯到autoencoder这种推荐模型中, 一直到BERT的突破, 有推荐模型像Cloze-like task一样进行预训练. 由于优化目标不同以及数据类型不同, NLP, CV的SSL 设计用于SSR并不兼容

作者将SSR的重要特点如下, 用pre-train并不算是对比学习, 只有对源数据做了数据增强才是, 那些没做数据增强以及仅仅优化marginal loss也不能当作SSR.

1. 获得更多的监督信号, 通过原始数据的半自动生成(最基本的, 表示SSR使用的范围)
2. 有pretext task 增强数据去(预)训练一个模型(说明了SSR的设置，使SSR有别于其他推荐)
3. 推荐任务依旧是主要任务, pretext task是辅助推荐的(指出了推荐任务与借口任务之间的关系)

所有的模型都可以简单的理解成是一个Encoder+Projection-Head架构, projection head 将学习到的特征$H$ 用于推荐或者其他特定的pretext task.

首先作者根据pretext task的特点将SSR模型分成四类, contrastive, predictive, generative and hybrid

**contrastive**: 最主要的办法, 核心思想就是将每一个instance看作为一个class, 然后将一样的instance拉近, 不一样的远离. 一个instance的两个视角可以看作是一个正样本对, 不同的instance被认为是负样本.

![image-20220906113130008](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906113130008.png)

![image-20220906113113867](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906113113867.png)

**generative**: 它的pretext task是去根据data的噪声版本去重现源数据, 主要目标就是预测原图片缺失的部分, 可以看作是一个自预测.

![image-20220906113631739](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906113631739.png)

![image-20220906113359106](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906113359106.png)

**predictive**: 新的samples或者标签总源数据中生成, 用来指导pretext task, predictive可以分为**sample-based和pseudo-label-based**, 前者注重更具当前状态的encoder预测informative samples然后将这些样本再送回到encoder中生成更高置信度的样本, 这种self-training也有数据增强, 能和SSL有联系. 后者通过生成器生成标签，生成器可以是另一个编码器或基于规则的选择器。然后，生成的标签被作为真实样本来指导encoder.

![image-20220906120600312](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906120600312.png)

**hybrid**: 有多个encoder做pretext task, 不同类型的pretext task通常将不同的自监督损失加权求和.

![image-20220906120632826](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906120632826.png)

接着作者根据训练方案分成三种, **Joint Learning(JL), Pre-train Fine-tune(PF), Integrated Learning(IL)**,也就是当有pretext task了之后应该怎么训练

**JL**: 最主要方法,  这种办法一般有一个共享的encoder来除了pretext task和推荐, 将结果分别计算两种Loss, 虽然JL可以被认为是一种多任务学习, 但是pretext task一般就是作为一个辅助的任务.

![image-20220906122658925](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906122658925.png)

![image-20220906122318883](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906122318883.png)

**PF**: 第二流行的, 首先会在有增强数据的情况下进行预训练得到较好的encoder初始参数, 然后将这些参数在原始数据上进行微调(无增强)然后通过一个projection head进行推荐. 还有一种办法是预训练一个无监督特征学习, 冻结encoder并且只学习一个linear head用于下游任务. 一般都是BERT-like SSR model,

![image-20220906123326747](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906123326747.png)

![image-20220906123126950](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906123126950.png)

**IL**: 很少用, 相当于在**JL**基础上只使用一个loss, 这个loss一般是用来计算互信息的, 一般IL用于pseudo-labels-based predictive method.

![image-20220906123510238](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906123510238.png)

作者接着介绍了一些基于序列的数据增强和基于图的数据增强办法.见文知意. 额外需要知道feature-based augmentation.  Feature Clustering是一个比较重要的办法, 通过EM办法聚类, 这种办法可以用于意图分类

接着作者开始详细介绍每一种之前提过的SSR方法

## CONTRASTIVE METHODS

![image-20220906153908807](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906153908807.png)

### Structure-Level Contrast (local代表node, item, global代表graph, sequence)

#### Same-scale ( Local-Local Contrast)

常用于图SSR 用于最大化user/item的源数据和数据增强后的互信息. 公式如下, 一般用的是node dropout这些办法, 有SGL, DCL, CCDR, PCRec, HHGR![image-20220906154631973](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906154631973.png)

#### Same-scale ( Global-Global Contrast)

常用于序列推荐. 将序列的两个增强后的结果通过Agg合成一个完整序列特征,

![image-20220906155920543](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906155920543.png)

#### cross-scale ( Local-Global Contrast)

用于将全局和局部的隐语义统一, 常用于图学习的场景.

#### cross-scale (Local-Context Contrast)

常用于图和序列推荐中, 所谓上下文常使用sampling ego-network or clustering

![image-20220906162100293](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906162100293.png)

### Feature-Level Contrast

比起Structure-level contrast研究较少, 因为数据集中可能没有支持, 一般工业界会用

### Model-Level Contrast

前两类从数据的角度提取自我监督信号，并没有以完全端到端方式实现。另一种方法是保持输入不变，并动态修改模型体系结构，以便动态地增强视图对。

### Contrastive Loss

对比损失的目标是u最大化多个视角的MI, 公式如下, 这个值无法直接计算, 需要最大化互信息下届, 为此有**JS Estimator和NCE**

![image-20220906165150327](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906165150327.png)

JS在[DeepInfoMax](./DeepInfoMax.md)有出现, 公式如下

![image-20220906165423970](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906165423970.png)

NCE Loss在[CPC](./Representation_Learning_with_Constrastive_Predictive_Coding.md)中出现, 公式如下

![image-20220906165554691](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906165554691.png)

有一篇论文研究NCE发现其中最重要的两个属性是alignment和uniformity

![image-20220906172835812](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906172835812.png)

### Pros and Cons

虽然没有报告表明对比SSR比其他SSR范式具有压倒性的优势，但它在利用轻量级架构改进推荐方面显示出了显著的有效性.一些常见的增强方法被认为是有用的，最近甚至被证明对推荐性能有负面影响

## GENERATIVE METHODS

主要MLM的generative SSR, 通过污染源数据去得到自监督信号, 分为Structure Generation 和Feature Generation

![image-20220906182124626](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220906182124626.png)

### Structure Generation

常用于序列推荐, 通过将原来的数据结构使用masking/dropout等数据增强操作后恢复成源数据, 参考BERT4Rec, 也可用于图, 就是想要恢复图原来的数据

![image-20220907095417919](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220907095417919.png)

### Feature Generation

图的情况下就是通过周围节点的feature, 预测一部分被掩盖的物品的feature. 序列的情况下就是预测未来的序列特征

![image-20220907100017405](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220907100017405.png)

### Pros and Cons

这类办法大部分都是followNLP的masked language models, 但是计算复杂度巨大

## PREDICTIVE METHODS

predictive SSR 完全通过原始数据通过self-generate得到监督信号. 分为**Sample Prediction, Pseudo-Labels Prediction**

![image-20220907105758493](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220907105758493.png)

### **Sample Prediction**:

self-training 和SSL相互联系. 这里的SSR model 首先在源数据上预训练, 然后基于预训练的参数让模型预测sample, 使用这些新的sample作为增强数据去完成推荐任务, 并且递归得生成更加好的sample. self-training和SSL之间的区别是自训练作为半监督学习只有有限的unlabeled sample而SSL中sample是动态生成的.

### Pseudo-Label Prediction

伪标签以两种形式呈现：预先定义的离散值和预先计算/学习的连续值. 前者通常描述两个对象之间的一种关系, 相应的pretext task目标是预测给定的一对对象之间是否存在这种关系. 后者通常描述给定对象的属性值或者特征向量, 相应的pretext task旨在最小化输入和预先计算的连续值之间的差异, 在这伪标签预测还可分为 **Relation Prediction, Similarity Prediction**

#### Relation Prediction

可以看成是一个分类问题, 将预先定义的关系作为伪标签, 受到BERT他和next sentence prediction的启发的

![image-20220907111119097](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220907111119097.png)

#### Similarity Prediction

可以看成是一个线性问题

![image-20220907112939584](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220907112939584.png)

### Pros and Cons

比起前面的predictive method生成sample更加灵活和动态.然而，作者认为也应该对使用预增强的标签持谨慎态度. 大多数现有的方法都是基于启发式方法收集伪标签，而没有评估这些标签和预测任务与推荐的相关程度.考虑到用户与项目的互动以及相关的属性/关系产生的理由（例如social）,有必要将专家知识作为先验加入到伪标签的收集中，这增加了开发预测性SSR方法的费用.

## HYBRID METHODS

通过多个pretext task来生成更多的sample, 有很好的result但是如何让多个pretext task合作是一个问题, 如何平衡不同数据增强之间的关系, 不同的数据增强结果之间相互会有关联. 繁琐的手工或者高昂的领域知识以及超参数的人工查找, 除此以外需要超高的训练华为

![image-20220907113840828](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220907113840828.png)

## DISCUSSION

数据增强的理论, 首先CV和NLP的增强办法因为数据和场景紧密结合, 没法直接用于RS, 现在的数据增强办法一般都是启发式的, 通过繁琐的尝试和试错(cumbersome trial-and-error)

自监督推荐的可解释性, 预训练模型的攻击和防御, 移动设备, 更加统一的预训练(无关乎推荐类型是电影 图片 文本)

