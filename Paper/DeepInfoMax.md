+ ***[Learning deep representations by mutual information estimation and maximization(DeepInfoMax)](https://arxiv.org/abs/1808.06670)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081315285.PNG)

**年份**：2018  

**引用次数**：1099  

**应用领域**：the basement of mutual information, unsupervised learning    

**方法及优缺点**：  

本文通过最大化输入和输出之间的互信息MI(mutual information)来无监督学习representation, 并且作者发现结构很重要, 整合输入中的局部信息(local MI)可以显著提高下游任务的适用性. DIM同时考虑了global MI, local MI, 和使用AAE(Adversarial autoencoders)思想限制先验概率,来约束学习到得representation, 让它得分布更容易处理.

**结论**：
在前人的基础上允许包含跨结构的局部一致性的特征表示, 换句话说就是新引入了local MI,让图片的每个分割的块都能保持MI一致性.

**动机**:  
作者在MINE的基础上发现,依赖下游任务类型的情况下,最大化输入和特征的MI学习不到有效的特征(global MI), 相反,结构很重要, 最大化特征和局部图片的平均值反而效果更好(local MI).但是global MI在重建任务上表现得很好.再然后根据AAE得思想,将MI和先验匹配结合起来约束学习到得representation, 让它得分布更容易处理.作者将此称为DIM(DeepInfoMax)

**相关工作和理论**： 
1\) 提出了 Deep InfoMax(DIM)，可以同时估算和最大化输入数据和高级representation之间的互信息(MI)  
2\) 作者提出的最大化互信息的方法，可以根据下游任务是分类还是重建，来对优化全局还是局部的信息进行调整  
3\) 使用AAE思想约束representation，来使其具有特定于先验的期望统计特征  
4\) 介绍了两种测量 representation 质量的新方法，一个基于Mutual Information Neural Estimation(MINE), 一个基于neural dependency measure(NDM), 并用它们将DIM与其他无监督学习方法进行比较  

对于使用一个生成模型, reconstruction error可以和MI通过下面这个公式关联, 其中$H$为熵, $R_{e,d}$为重建误差, 通过减小$R_{e,d}$提高了MI的上确界, 以此来提高MI.  
$I_e(X, Y) = H_e(X) - H_e(X|Y) \geq H_e(X) - R_{e,d}(X|Y)$
对于MI计算估计也有许多前人的工作, 作者提到了以及后面用到了JSD散度, 更稳定并且效果很好, 作者后续会结合MINE和JSD来做MI估计完成4)的内容.以及作者提到了CPC是独立于DIM提出的, 类似两个门派, CPC和DIM有共同的动机和计算方法, 但是CPC是按照顺序处理局部特征, 用于预测每个局部特征的"未来", 这需要训练独立的估计器来预测不同的"未来"的step偏移量, 该特征的学习会极大得提高计算复杂度, 但是DIM于此相反, 只训练一个单一的特征提取提取出全局的特征并且通过这个全局特征来对自我的局部特征做无序预测.   

接着开始介绍DeepInfoMax   
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081315707.PNG)  
$x$ 表示原始图像， $f_\psi(x)$表示经过卷积层输出的$M \times M$的 feature map，而 $y=h_\psi (f_\psi(x))=E_\psi{(x)}$则为经过全连接层之后最终的 feature vector（也可以理解为图像的一种 representation, 论文中作者说的是把feature map的求和得到），再将这个 representation 应用于一些下游任务. 
而 DIM 的整体思路如下图，也就是用图片encode后得到的representation, 与同一张图片的 feature map 组成正样本对，和另一张图片的 feature map 组成负样本对，然后训练一个 Discriminator 来区分这两中样本对.  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081315575.PNG)  
DIM的终极目标就是两点

- 最大化$f_\psi(x)$和Y也就是feature map和feature vector之间的MI,这又分为global和local两种
- 在最终的 representation 中再加入一个统计性的约束，使得到的$y$的分布（push-forward distribution） $Q_{\psi,X}$尽量与先验分布$Q_{prior}$相匹配

对于1)中最大化MI,就需要写出loss, 对于$X,Y$如果他们相互独立则互信息为0,并且$P_{XY} = P_XP_Y$, 互信息的定义如下  
$I(X;Y) = \sum_{X,Y}P(X,Y)log\frac{P(X|Y)}{P(X)}$ 并且由此可以推得下图得KL散度公式, 可以用来衡量联合分布和独立分布得关联性, 所以互信息越大越好.
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081316729.PNG)  
首先介绍global MI的情况,互信息没法精确计算,但是有一些方法对其进行了估计, 比如MINE找到了其下界$\hat{I}^{(DV)}_\omega$,$P(X,Y)$为来自同一张图片, $P(X)P(Y)$为来自不同图片, 所以要最大化这个下界就要最大化前者, 最小化后者   
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081316203.PNG)  
所以问题转换为了最大化 $\hat{I}^{(DV)}_\omega$ 这个下界的问题, 所以可以写出我们的loss function,如下图所示,完成1)的目标   
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081316105.PNG)  

但是我们想要优化目标,并不需要知道MI的具体数值, 只要将其MI最大化即可,所以上式可以写成  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081317394.PNG)  
其中的$\hat{I}_\omega$可以换成其他的MI估计比如JSD和InfoNCE, 如下图, 从附录的结果来看JSD 几乎不受负样本数量的影响，而 InfoNCE 的效果则随着负样本数量的降低而降低，DV 受到负样本数量的影响最大，但是随着负样本数量的增加，他们之间的差距会逐渐缩小。  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081317148.PNG)  
至此总结global informax的流程, 如下图所示    
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081317511.PNG)  

对于一张图片来说，如果我们的下游任务不是重建类的任务，只是对图片进行分类，那么就没有必要对某一些琐碎或者对分类任务无关紧要的像素级噪音进行编码。而如果我们设定的目标是”最大化整张输入图片的 feature map 与 representation“，那么实际上会无法控制究竟哪些部分传入了编码器. 而Local Infomax 的思想就是，我们并不将整张图片的 feature map 一次性输入损失函数来进行MI最大化, 而是将其分块,最终目标是使这$M^2$个块和整张图片的 representation 的平均 MI 达到最大，这样就使最后的 representation 和每一块的 MI 都达到最大，从而达到对每个块之间共享的一些信息进行编码的效果, 所以这能很简单的写出local infomax的loss公式 
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081317376.PNG)   
至此总结local informax的流程,如下图所示,主要是在第(3)(5)步中，现在需要对每个patch执行此操作,并且这个j是无序的,再哪一行哪一列并不重要.    
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081318998.PNG)  
原理图如下图所示  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081318055.PNG)  
至于这么计算这个平均互信息, 附录给出了两种方法
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081318491.PNG)  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081318683.PNG)  

接下去要限制先验概率,因为
若学习到的隐变量服从标准正态分布的先验分布，这有利于使得编码空间更加规整，甚至有利于解耦特征，便于后续学习。因此，在 DIM 中，作者同样希望加上这个约束.  鉴别器的目标是区分 representation 分布的真伪（即是否符合先验分布），而编码器则是尽量欺骗判别器，输出更符合先验分布的 representation.  
具体做法是,训练一个鉴别器$D_\varPhi$, 我们需要学习到一种representation,来让这个鉴别器$D_\varPhi$, 确信其是否来自先验分布$Q_{prior}$  
鉴别器的损失函数如下所示(这里有待深化理解),作者在实验中发现这里用均匀分布比高斯分布效果更好  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081318464.PNG)  

最后DIM的最终目标就是把上述三个目标放到一起, 如下:  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081318736.PNG)
之所以加上$\alpha,\beta,\gamma$三个参数，是因为有时候我们只想使用 global InfoMax (如重建类下游任务)，就可以将$\beta$设置为0；而有时候只想使用 Iocal InfoMax (如分类任务)，就可以将$\alpha$设置为0；但这两种情况下，最佳的$\gamma$值不同的.

**实验结果**:  
复现的时候回来补

**个人总结**： 
个人感觉在较新得MI学习representation论文, 理解得差不多了,但是一些基础理论, 比如更前面得ICA,MINE, NDM, NAT,甚至是更早的infomation-maxization相关的内容. 而且之前看到一篇文章讲起MI的内容还出现了F-GAN的内容.

**备注**:  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081319072.PNG)
[KL散度百度百科](https://baike.baidu.com/item/%E7%9B%B8%E5%AF%B9%E7%86%B5/4233536?fromtitle=KL%E6%95%A3%E5%BA%A6&fromid=23238109&fr=aladdin)
