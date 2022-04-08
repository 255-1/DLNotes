+ ***[Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)***   
  

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081319779.PNG)  

**年份**：2020  

**引用次数**：2849  

**应用领域**：unspervised learning, contrastive learning  

**方法及优缺点**：    
Moco将对比学习特征看成一个字典查找问题, 并且总结了前人的问题, 通过使用队列+动量编码器实现了一个又大又一致的的字典办法, 前人的成果只能有其中一种办法.这些特征在下游任务上效果很好.

**结论**：
Moco在许多cv的任务和数据集上展现出很好的结果, 但是Moco在1M到1B的数据的提升上很少, Moco需要一个更好的pretext task让Moco在大数据集上有待充分利用.  

**动机**:  
作者认为无监督学习在nlp领域效果好是因为语言任务是在一个分离的信号空间中, 但是cv的原始信号是在一个连续的高维的空间中, 并不是为了人类交流而构建的.接着作者提到了无监督表征学习可以看成是一要建立一个动态字典.  
所以作者建立了一个又大又一致的字典, 并且Moco可以和许多pretext task一起做, 本文使用的[InstDisc](./Unsupervised_Feature_Learning_via_Non-Parametric_Instance_Discrimination.md).

**相关工作和理论**： 
作者希望Moco实现的目标:  
1\) 动态字典足够大  
2\) 字典内容一致性  

Loss function使用的时[CPC](./Representation_Learning_with_Constrastive_Predictive_Coding.md)中的InfoNCE,如下图所示, 这里不过多赘述.  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081319549.PNG)  
接着,作者认为对比学习需要足够的负样本, 之前知乎看的别人的推导, 当负样本足够大时NCELoss才能逼近MLE的结果, 最近我复现CPC的时候也发现了负样本不够多时模型一点都不work. 所以这里使用队列的办法, 让字典大小不再受限于一个mini-batch的大小, 每次都可以把最老的一段移除出去加入最新的, 字典大小可以自行设置,由此完成1).   
但是又出现了一个新问题, 这个队列没法通过反向传播整个更新, 因为每次只会更新一个mini-batch大小,但是我们应该更新整个队列的内容, 缺乏一致性, 所以作者引入动量更新的办法解决2). 如下图所示, 每次只有query的编码器需要更新, 而key的编码器通过设置动量变量$m$(默认=0.999)控制,这让key的编码器的变化更加平滑.    
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081320749.PNG)   
接着作者对比了end-to-end, memory bank以及Moco,很明确的表现了Moco和前人的不同之处, 如下图所示, end-to-end办法能保持一致性但是字典太小, memory bank字典可以很大, 但是每个epoch才会更新一次, 所以一致性太差了, Moco不仅能有更好的内存效率而且能在十亿级数据上训练.  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081320466.PNG)
最后作者说,key和query都有Batch Normalization(BN),但是这个BN会影响模型的效果,作者认为是pretext task泄露了信息, 让model找到了捷径.作者使用的时shuffling BN(现在还不太理解),前提是多GPU, 每个GPU独立对样本执行BN, 原文内容如下所示.  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081320600.PNG)  
最后就是Moco算法伪代码,如下图所示, 所使用的是[InstDisc](./Unsupervised_Feature_Learning_via_Non-Parametric_Instance_Discrimination.md) pretext task  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081320607.PNG)

**实验结果**:  
复现回来补

**个人总结**： 
在看过前置文章之后, Moco理论并不复杂, 简单且高效.

**备注**:  

