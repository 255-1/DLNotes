+ ***[Contrastive Multiview Coding(CMC)](https://arxiv.org/abs/1906.05849)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081305185.PNG)  

**年份**：2019  

**引用次数**：841  

**应用领域**：the basement of constrastive learning, unspervised learning   

**方法及优缺点**：
通过最大化同一物品的不同视角下的mutual information学习到的特征更加好, 视角越多效果越好, 作者推广了之前的只有两个视角的方法到更多视角的情况([CPC](./Representation_Learning_with_Constrastive_Predictive_Coding.md), DeepInfoMax, [InstDisc](./Unsupervised_Feature_Learning_via_Non-Parametric_Instance_Discrimination.md)).   

**结论**：
本文提出了一个对比学习的无监督表征学习的框架.构建的准则是最大化互信息学到强力的特征.并且比起predictive learning表现更好, 并且随着视角的增加有效性在增加.  

**动机**:  
对于autoencoder来说, 每个bit是平等的, 但是作者认为有些bit在语义,物理意义和几何意义上比起其他bit更加重要. 作者认为一个好的bit是在不同视角下共享的部分,并且能让下游任务完成得更好.比如在Lab色彩模式中就可以把一张图看成一个L视角和ab视角下的成对实例.作者在CPC的基础上做出了简化和推广,去除了自回归部分的RNN并且推广了多视角的情况.
       
**相关工作和理论**： 
作者认为自己对[CPC](./Representation_Learning_with_Constrastive_Predictive_Coding.md), DeepInfoMax, [InstDisc](./Unsupervised_Feature_Learning_via_Non-Parametric_Instance_Discrimination.md)的改进:  
1)  扩展到2个视角以上了   
2)  对不同的视图定义, 体系结构和应用程序设置  
3)  对表征学习方式进行了独特的研究  

首先介绍predictive learning(如下图的上半部分)如autoencoder, 它希望最后输出的 $\hat{v_2} $ 能尽可能的接近$v_2$,这种目标提前假定每个像素或者元素之间相互独立,并且 loss是在output空间上做, 并不能直接优化特征$z$, 常用的loss function(如L1, L2)是非结构性的, 而对比学习的loss是直接作用在特征空间中, 直接对特征的一致性和非一致性进行学习.所以认为学习到的特征的效果会比predictive learning好. 
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081305142.PNG)  

作者接下来首先介绍的是双视角情况下的计算,作者认为样本$x \backsim p(v_1, v_2)$或样本对 $x={\{v_1^i, v_2^i\}}$为正样本, 对于$y \backsim p(v_1)p(v_2)$或样本对$x={\{v_1^i, v_2^j\}}$为负样本, 应该是认为相关的内容为正,如果是独立分布的情况下就认为是负样本, 这样可以让互信息尽量大. 设定判别标准为$h_\theta(\cdot)$, 样本为$S={x, y_1, y_2, ....,y_k}$,则loss function如下图所示, 具体理解在之前的[InstDic](./Unsupervised_Feature_Learning_via_Non-Parametric_Instance_Discrimination.md)有过,这里看着很好理解. 
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081306560.PNG)  
将两个视角中的一个视角固定为anchor不变,则公式可以写成如下形式,但是下面这k可能会非常的大, 所以需要使用NCE简化.  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081306928.PNG)   
作者定义了一个简单的$ h_\theta(\cdot) $,如下图所示  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081306097.PNG)  
然后我们可以互换anchor,并将两个视角的Loss的和作为双视角的Loss结果, 公式如下图所示  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081306207.PNG)  
然后作者解释了这个为什么可以和互信息关联,其实在之前[CPC](./Representation_Learning_with_Constrastive_Predictive_Coding.md)那片文章里面已经写过了,这里理解起来也蛮快的,为什么$h_\theta(\cdot)$会成比例以及为什么互信息下界的证明,如下图所示,本文在附录里面也有写,这里不过多赘述,但是作者最后提到了在一篇最近的论文中显示这个下界的证明可能很弱, 所以需要找一个更好的互信息评估的公式.  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081307839.PNG)  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081307172.PNG)  

接下来作者就将刚刚的双视角推广到多视角的情况下, 提出了更加广泛的Loss Function公式(如下图所示),其实就是按照不同anchor求出来后再整体求和.但是按照anchor计算(core view)还是整体计算(full graph)是要在有效性和有效率上面权衡的.  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081307562.PNG)
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081307489.PNG)

上面提到的"core view"和"full graph"两种新范式如下图所示,上面两个公式都表明了信息的优先度和共享该信息的试图数量成正比.对于左边的"core view",我对图片理解是要计算$v_1$与$v_2$,$v_1$与$v_3$, $v_1$与$v_4$, $v_1$与$v_2$$v_3$,$v_1$与$v_2$$v_4$,$v_1$与$v_3$$v_4$,$v_1$与$v_2$$v_3$$v_4$, 与后者的比较数量就为图片上的数字. 对于"full graph"则需要更加全面的信息, 对于右边的图的数字的计算就是要按照区域被几个View给覆盖,从中取两个做对比,比如最中间为6, 因为被4个View覆盖,所以$C_4^2=6$, 同理3被3个View覆盖而$C_3^2=3$, 数字越大代表互信息的权重更大,更重要.   
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081308815.PNG)  

作者最后认为"full graph"能得到更加不同视角下的信息,可能对于下游任务更加有用,并且"full graph"能以更加自然的方式处理原本丢失的信息.  

最后Contrastive Loss的NCE近似计算之前k过大问题以及memorybank的使用和以前的InstDisc以及CPC类似,作者附录也有证明,这里不过多赘述.

**实验结果**:  
复现的时候再回来补.  

**个人总结**： 
CPC,个体判别都和这篇文章很像,但在视角的选取上又有所不同,比如CPC学习过去和未来两个视角,个体判别呢又是学习一张图片的不同crops.总之过了CPC的基础再看这篇CMC就轻松很多了.

**备注**:  

$$
