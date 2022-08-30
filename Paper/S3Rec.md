+ ***[S3-Rec](https://dl.acm.org/doi/abs/10.1145/3340531.3411954)***   

![image-20220830170121834](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220830170121834.png)

**年份**: 2020

**引用次数**： 160

**应用领域**：对比学习推荐系统  

**方法及优缺点**：

普通的序列推荐模型容易受到数据稀疏问题. 所以提出了本文的S3Rec.该方法的主要思想是利用内在的数据相关性来获得自我监督信号，并通过预先训练的方法来增强数据表示，以改进序列推荐.  使用了自监督目标和MIM来学习不同的属性,物品, 子序列和序列这些的关联. MIM可以提供一个统一的方法去表现数据的关联性. 这是第一个序列推荐的pre-train的序列推荐模型, 自监督目标为**item-attribute,sequence-item, sequence-attribute和sequence-subsequence**

**动机**:  

普通的序列推荐模型当合并上下文数据时，所涉及的参数也通过唯一的优化目标来学习。研究发现，这种优化方法容易受到数据稀疏问题影响. 其次, 他们十分重视最后的performance, 导致上下文的关联和信息融合并不能被很好的捕获, 学习到的特征太狭隘. 自监督学习作为一个新出现的框架解决了上述的两个问题. 但是传统的序列推荐里面上下文信息里面包含不同的物品属性等信息, 很难有一个统一化的方法去描述, 所以打算使用MIM方法

**相关工作和理论**：  

[InfoNCE](./DeepInfoMax.md)的公式就是交叉熵公式的推广, 如下

![image-20220830174753757](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220830174753757.png)

![image-20220830174806909](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220830174806909.png)

可以针对不同的X,Y分布进行计算, 对于序列使用的基本的embedding+self-attention和预测层的堆叠, self-attention使用的是Bidirectional self-attention, 在pre-train阶段不使用mask, 在预测阶段使用mask transformer类似SASRec

**item-attribute**: 让序列的属性序列和原序列相近, 因为attribute可以提供细粒度的信息, 所以要让属性$\mathcal A_i={a_1,...,a_k}$中的每个属性都和原序列对应的id更加相似. 公式如下按照上面的交叉熵公式改的.
$$
L_{AAP}(i,\mathcal A_i) = \mathbb E_{s_j\in\mathcal A_i}[f(i, a_j)-log\sum_{\tilde a\in\mathcal A \backslash \mathcal A_i}exp(f(i, \tilde a))] \\
f(i,a_j)=\sigma(e_i^T\cdot W_{AAP}\cdot e_{a_j})
$$
**sequence-item**: 就是mask一个item ,让被mask的item和原序列相近, 就是NLP的BERT的Cloze task.  将被挖掉的周围的context 定义为$C_{i_t}$, $F$是序列特征. 

![image-20220830175035936](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220830175035936.png)

![image-20220830175045390](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220830175045390.png)

 **sequence-attribute**: 和上面的类似, 不过不是item而是item的属性

![image-20220830180615006](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220830180615006.png)

![image-20220830180624014](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220830180624014.png)

**sequence-subsequence**:相当于将前面的Cloze task从一个item拓展到item sequence, 比起单个的item更加的稳定,

![image-20220830181026445](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220830181026445.png)

![image-20220830181035172](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220830181035172.png)

模型训练过程为pre-train和fine-tuning, 预训练时就是双向Transformer和上面的四个目标, 然后把学习到的参数初始化赋值给微调参数, 然后从左到右的训练. fine-tunning的pairwise rank loss公式如下
$$
L_{main}=-\sum_{u\in\mathcal U}\sum_{t=1}^nlog\sigma(P(i_{t+1}|i_{1:t})-P(i_{t+1}^-|i_{1:t}))
$$


**实验结果**:  

复现回来补

**个人总结**：  

**备注**  