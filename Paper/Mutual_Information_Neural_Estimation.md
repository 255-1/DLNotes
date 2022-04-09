+ ***[Mutual Information Neural Estimation(MINE)](https://arxiv.org/abs/1801.04062)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081321550.PNG)

**年份**：2018  

**引用次数**： 447  

**应用领域**：mutual information  

**方法及优缺点**：  
作者通过找到了一种办法可以通过神经网络计算高维连续的变量之间的互信息, 这种新的办法在维度和样本量上都是线性可伸缩的, 可以通过back-prop进行训练并且有强一致性.  

**结论**：
首先引入互信息可以缓解GAN中模式丢弃问题, 互信息还可以用于改进ALI的推理和重建, 最后MINE已于处理信息拼劲问题.  

**动机**:  
对于互信息的定义没有变, 如下图所示, MI可以抓取变量间的非线性的统计性依赖,但是这个MI公式难以直接计算  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081321170.PNG)  
所以本文是基于一个KL散度近似计算MI, 通过计算联合概率分布和边缘概率分布的KL散度的对偶问题等价于MI的值,具体推导如下图所示, 展示了MI和熵和KL散度之间的等级关系的推导    
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081321411.PNG)  
作者也使用了F-GAN中的f散度,这个散度在没有任何明确的对数据分布的假设的情况下训练生成模型.  

**相关工作和理论**： 
互信息和KL散度之间的关系在上一节已经展示.将KL散度中的联合概率分布写为$P$,边缘概率分布的乘积记为$Q$,将log前的$P$提出来作为期望的抽样方式,则KL散度为下图所示.  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081321920.PNG)  
由之前的等式可以看出上式的KL散度越大,则$X$和$Z$之间的MI就越大.  
所以现在互信息的计算问题等价于计算KL散度的问题, 而KL散度的计算通过一个对偶问题计算DV散度可以得到其下界.  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081321069.PNG)  
至于为什么右边这个公式的上确界为KL的下界, 在附录中的推导如下图所示, 引入玻尔兹曼分布$g(x)=\frac{1}{Z}e^{T(x)}q(x)$也就是$dG=\frac{1}{Z}e^{T}dQ$, 另外$T$为一个判别器完成$\Omega\to R$, T就为一个DNN,作者称这个网络为statistics network  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081321376.PNG)  
则KL和上式之间的gap为他们的差值,  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081322228.PNG)  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081322627.PNG)
这意味着什么呢,上式=0也就是下界等于KL散度当且仅当P和G的分布一致, 然后$dP=dG=\frac{1}{Z}e^{T}dQ$, 然后将T移到一边得$T=log\frac{dP}{dQ}+C$, 所以可以反向推得我们优化一个T就可以最小化KL散度和下界之间得gap,而这个优化T就可以交给一个神经网络来做.  
暂时总结一下就是可以通过训练一个神经网络得到一个足够好得判别器$T^*$,这个过程可以减小KL散度的值和下界之间的gap,假设最优的情况下,gap=0则这个估算值也就是两个分布之间的互信息的估算数值.   
作者接着引入了一个比DV弱一点的下界,f散度  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081322260.PNG)  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081322613.PNG)  
接下去作者先引出neural information measure,其中的期望估计可以使用$P_{XZ}$和$P_XP_Z$的抽样或者是在batch上的联合概率分布的shuffling the samples.所谓从联合概率抽样,感觉很抽象, 我根据InstDisc中的了解大致理解是, 什么是联合概率是由人定义的正样本,类似一个pretext task,比如InstDisc中认为一张图片和自己的数据增强为联合概率分布,这个很直觉, 而与之相对的从边缘概率抽样我的理解就是认为定义的互信息不足的办法.  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081322282.PNG)  
接着作者就正式讲解MINE,
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081323080.PNG)  
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081323413.PNG)  
具体优化过程为本文最上面的算法图中的内容.作者在这里提到了在mini-batch中使用SGD得到的MINE的结果是biased, 可以将下图导数的分母用exponential moving average(指数移动平均?), 这能改善性能
![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081323468.PNG)  
接着作者又分析证明了一致性和收敛性以及抽样复杂度,扫了一眼好像都在附录里面,现在对我来说有点硬核, 以后再看.
**实验结果**:  
复现再回来补

**个人总结**： 
本文讲解的主要是能通过神经网络来计算互信息的方法,个人感觉互信息看到这里也就差不多了,再要往早追述有点得不偿失,需要找时间再刷一遍之前看过的论文,串一串内容


**备注**:  
[Gibbs分布](https://en.wikipedia.org/wiki/Boltzmann_distribution)  
[exponential moving average](https://blog.csdn.net/qq_14845119/article/details/78767544)
