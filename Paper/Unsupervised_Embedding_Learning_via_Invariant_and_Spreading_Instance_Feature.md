+ ***[Unsupervised Embedding Learning via Invariant and Spreading Instance Feature](https://openaccess.thecvf.com/content_CVPR_2019/html/Ye_Unsupervised_Embedding_Learning_via_Invariant_and_Spreading_Instance_Feature_CVPR_2019_paper.html)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081325835.PNG)   

**年份**: 2019  

**引用次数**：231  

**应用领域**：cv, unsupervised learning, the basement of constrastive learning  

**方法及优缺点**：  
传统的无监督学习学习到的所谓的"intermediate"特征可能无法保持视觉上的相似性,基于相似性的任务性能会急剧下降.本文提出的Siamese架构旨在学习到数据增强的Invariant和个体的spread-out特征.并且直接通过优化最顶层的softmax函数达到快速学习并且高准确率的目的.在训练测试集没见过的任务上也表现得十分良好.  

**结论**：
在这片论文中提出了学习Invariant和spread-out特征,通过拉近同一张图片在不同数据增强下的数据得到Invariant, 扩大同一个batch中的不同个体的特征数据得到spread-out,训练本文提出的Siamese架构,结果好效率高. 并且我们从经验得出 spread-out属性在实现视觉相似的情况下是非常重要的.  

**动机**:  
传统的无监督的特征学习在视觉相似性或者无标签数据中表现得不理想. 作者希望能让学习过程只依赖于instance-wise的关系, 而不是来自预先人为定义的类别.  

**相关工作和理论**： 
在传统的无监督特征学习中有三种办法:
1)生成模型,旨在学习图像和预定义噪声之间的映射,从而限制原始数据和噪声之间的分布.  
2)聚类办法   
3)自监督学习,定义一个pretext task去生成虚伪的label  
但是以上办法得到的视觉相似性效果不好.  
传统的Embedding学习可以分为:  
1) Deep Embedding Learning一般要使用抽样策略比如hard mining, semi-hard mining, smart mining. 而本文不需要这些策略,使用softmax embedding即可.   
2) 无监督Embedding学习主要分为训练集和测试机类别相同,训练集和测试机的类别不同.这解都非常依赖与标签挖掘的初始化表征.  

前人做过的事情.  
首先是softmax embedding with Classifier Weights的缺点,使用分类器权重无法明确和特征$f$比较,导致效率低可辨别性低.   
$P(i|x_j) = \frac{exp(w^T_if_j)}{\sum_{k=1}^n{exp(w^T_kf_j)}}$ (Eq.1)  
接着是[InstDic](./Unsupervised_Feature_Learning_via_Non-Parametric_Instance_Discrimination.md)中的Softmax Embedding with Memory Bank的缺点.虽然解决了特征比较的问题,但是效率低下,$x_i$对应的memory bank中的特征$v_i$仅仅在输入$x_i$的时候更新,换句话说就是$v_i$每个epoch才更新一次,但是网络是每个iteration都会更新,更新滞后会组织训练进程.  
$P(i|x_j) = \frac{exp(v^T_if_j/\tau)}{\sum_{k=1}^n{exp(v^T_kf_j/\tau)}}$ (Eq.2)  
综上是以前论文的一些问题,一种直觉的改进Eq2效率方案是将Eq.2中的$v_i$换成$f_i$,但是出于两种原因这种方法不可行:  
1)  对于$P(i|x_i)$会变成$f_i^Tf_i=1$无法更新网络权重, 没法优化.  
2)  如果instance太多Eq.2中的分母计算量太大  

基于以上原因提出了本文的softmax embedding on 'Real' Instance Feature  
对于不可行的原因1)不直接计算$P(i|x_i)$, 改为计算数据增强后的结果为i的概率,即$P(i|\hat{x}_i)$,这样就不会是1,是一个可以优化的过程, 通过架构中的共享权重可以实现.对于不可行的原因2)通过采样一个batch中的其他$m$个负样本而不是全部数据样本来简化计算.由此可以得到下面的计算正样本的概率:  
$P(i|\hat{x}_i) = \frac{exp(f^T_i\hat{f}_j/\tau)}{\sum_{k=1}^m{exp(f^T_k\hat{f}_j/\tau)}}$ (Eq.3)   
对于batch中其他负样本的概率就可以计算为  
$P(i|x_j)=\frac{exp(f_i^Tf_j/\tau)}{\sum_{k=1}^mexp(f_k^Tf_j/\tau)}, j\neq i$(Eq.4)    
假设每个instance之间是相互独立的,则$\hat{x}_i$被认为是instance的类型$i$并且$j\neq i$不被认为是类型$i$的联合概率分布为如下式子:  
$P_i=P(i|\hat{x}_i)\prod_{j\neq i}(1-P(i|x_j))$(Eq.5)  
对上面的式子求最大似然估计,损失函数可以写成negative log likelihood    
$J_i=-logP(i|\hat{x}_i)-\sum_{j\neq i}log(1-P(i|x_j))$(Eq.6)  
Eq6只是一个instance的计算,则整体的Loss function计算为:  
$J_i=-\sum_{i}logP(i|\hat{x}_i)-\sum_i{}\sum_{j\neq i}log(1-P(i|x_j))$(Eq.7)  
对于Eq.6中的之子前面的log里面为Eq.3, 后面的log里面为Eq.4的式子,最小化Eq.6需要最大化Eq.3最小化Eq.4从而完成了减小正样本的距离,增大和负样本的距离的目标.  

**实验结果**:  
等到复现的时候回来补  

**个人总结**： 
作者提出了一种不需要使用memory bank这种额外数据结构的办法, 对于个体判别任务的效率提升了很大. 但是依赖了同一个batch中的负样本数量, 根据InstDisc笔记中的备注内容可以知道, 负样本数量决定了NCE能不能很好近似极大似然估计结果, 同一个batch可能会让负样本数量太少.
**备注**:  

