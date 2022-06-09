+ ***[Session-based Recommendations with Recurrent Neural Networks(GRU4Rec)](https://arxiv.org/abs/1511.06939)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081324456.PNG)

**年份**：2016  

**引用次数**：1110  

**应用领域**：sequtial recommendations

**方法及优缺点**：

这篇文章是RNN首次使用在RS领域的文章,作者认为现实生活的RS通常只能是 short session-based数据. 这种情况下常用的矩阵分解技术并不准确, 在实际工程里这个问题通常通过推荐相似物品来解决. 作者通过对整个session建模可以得到更好的准确度, 并通过引入一些修改来让RNN适应任务.  

**结论**：

作者认为SBRS(session-based recommendation system)是一个重要但是缺乏研究的领域, 通过引入GRU到RS领域,以及作者提出的session-based mini-batches以及基于mini-batches的ranking loss function来修改GRU以更好的适应任务.  

**动机**: 

许多用户在较小的购物网站只有一两个session, 并且在某些领域中用户通常表现出session-based特征.许多的SBRS未完全使用用户行为画像比如只用最后一次click进行预测.  
作者认为常用的factor model只是分解出了特征,忽视了用户行为画像, 难用于SBRS,  所以使用了另一种常用的neighborhood method基于计算session中的物品或者用户相似度的一种办法.  
对于推荐系统的稀疏序列数据以及将RNN引入RS的问题,作者引入了新的ranking loss function解决.将用户的第一次点击作为RNN的输入,输出一个可能的物品,每一次后续的点击都会根据前面的所有点击得到推荐物品.  
主要面临的挑战有物品集合可能很大,点击流的数据集通常非常大.

**相关工作和理论**：

本文主要解决了三个问题:

1. 将RNN引入RS 
2. 点击流的数据集通常很大 
3. 物品集合可能很大   

​	对于SBRS常用item-to-item推荐算法只考虑了最后一次点击的行为. Markov Decision Processes方法如果要考虑用户所有的序列行为会无法计算. 扩展的General Factorization Framework方法将item的特征分为item自身的特征和作为session一部分的特征, 并且将session作为sessoin中每一个item的session特征的权重和,但是这种办法没有考虑session中的行为的先后关系.  

所以作者提出了一种基于GRU的办法,列出的公式都是来自GRU的设计,见备注. 使用one-hot编码, 将session中一第一个event的编码输入,输出就是session中的下一个event.由此完成了1). 

接着作者说RNN在nlp领域通常使用滑动窗口完成batch中的采样,但是在RS中有的session可能只有2个events,而有的session可能有上百个events. 我们的主要目标是学习到session是如何随着时间变化的,所以作者提出了session-parallel mini-batches,如下图所示, 将$X$个session的第一个event作为mini-batch的输入,而下一个输出作为mini-batch的输出,如果session结束了,就自动补入下一个session的第一个event.作者假设这些session之间是相互独立的.由此完成了2).    ![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081324773.PNG)    

接着作者对于item非常多,如果每一个需要评分需要花费大量计算,因此需要对于输出进行抽样并且只计算其中的一小部分.这会让网络只更新部分权重,所以也要计算negative exmaples的分数来更新权重.并且作者认为缺失的event就是用户不喜欢.综上我们要应该根据流行程度来抽样,并且将一个mini-batch中的其他样例作为negative examples(exmaple出现在一个mini-batch中的可能行和其流行程度成正比).由此完成了3).

最后作者提出了专门为RNN的RS的新的Loss function.基于pairwise的ranking loss. 

第一种是基于BPR的一种矩阵分解技术,比较了正样本和负样本的分数,$\hat{r}_{s,i}$为正样本打分,$\hat{r}_{s,j}$为负样本打分.

$L_s=-\frac{1}{N_s}\sum_{j=1}^{N_s}log(\sigma(\hat{r}_{s,i}-\hat{r}_{s,j}))$  

第二种是基于第一种的一种正则近似计算相关物品的相关排名的公式,去除了上面的log并且加入了$\hat{r}^2_{s,j}$为了让对于负样本的分数希望越低越好.

$L_s=-\frac{1}{N_s}\sum_{j=1}^{N_s}\sigma(\hat{r}_{s,i}-\hat{r}_{s,j})+\sigma(\hat{r}^2_{s,j})$ 

**实验结果**:  

复现回来补

**个人总结**： 

据说是RNN引入RS的开山作,现在对序列推荐连皮毛都不太清楚,还有待学习.

**备注**:  
[GRU讲解](https://www.youtube.com/watch?v=T8mGfIy9dWM&t=1952s)
