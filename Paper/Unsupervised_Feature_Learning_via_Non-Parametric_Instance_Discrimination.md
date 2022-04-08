+ ***[Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination](https://arxiv.org/abs/1805.01978)***   
  ![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081327381.PNG) **年份**：2018  

**引用次数**：1013  

**应用领域**：basement of constrastive learning   

**方法及优缺点**：
  无需数据标签的无监督学习,通过instance-level级别的学习,直接学习图片的特诊,在下游任务中通过fine-tuning也可以有很好的结果,并且此方法对计算复杂度和存储要求较低  

**结论**：
  同上  

**动机**:  
  通过观察有监督学习的学习结果的图片发现分类器认为是同一类的物体在视觉上也是十分相似的,就是比起不同类的物品长得相似.
  ![观察图片](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081325648.PNG)   

**相关工作**：  
  需要解决得问题有:  
  1)由于每张图片为一类,softmax计算归一化因子的复杂度太大  
  2)提炼出来的feature只使用SVM分类器不完善  
  3)提出一种新的non-parametric的评估方法,使用memory-bank而不是net的weight  

  对于1)提出了新的softmax公式,将权重$w$改为features $v$,并且让$||v||=1$, 同时引入温度超参数$\tau$用来控制分布的concentration level
  ![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081326728.PNG)    
  对于2)使用额外的KNN分类器,对于一张图片的$k$个邻近的对象, 计算属于哪个类.   
  对于3)使用了改进的使用了L2正则的NCE(也可以使用negative sampling), NCE基本思想是将多分类问题转化为一组二分类问题，其中二分类任务是区分数据样本和噪声样本.
  下图公式为计算正样本的概率,其中假设noise的分布$P_n=1/n$为均匀分布,并且假设noise samples are m times more frequent than data samples,  
  ![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081326061.PNG)    
  所以loss function就是要最小化下图的公式,其实就是要最大化$h(i, v)$, 其中$v'$为noise的特征, 所以$1-h(i,v')$就是正样本,也要最大化,  
  ![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081326197.PNG)  
  下图的公式为计算memory bank中$v$对应类型$i$的概率  
  ![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081326419.PNG)    
  由于$Z_i$的计算复杂度太大,使用蒙特卡洛近似成一个常量,其中${j_k}$是一个随机的小切片  
  ![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081326168.PNG)  
  由于每一个epoch只会用到instance一次,所以学习梯度比较陡峭,由此引入L2正则化让学习曲线更smooth,公式如下图所示,其中$v^{(t)}_i=f_\theta(x_i)$为当前data出来的feature representation, 而memory bank中存有之前的$V = {v^{(t-1)}}$  
  ![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081326324.PNG)  
  由此可以到的类似上面的NCE loss function如下图所示  
  ![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081326847.PNG)  

**实验结果**: 

复现回来补  

**个人总结**： 

对比学习Moco中的基础文章,算是了结了个体判别的Pretext work的方法以及一些关于NCE的知识   

**备注**  
1)[NCE计算推导](https://zhuanlan.zhihu.com/p/76568362/)  
2)[github](https://github.com/zhirongw/lemniscate.pytorch)  
3)当负样本数量趋于无穷时, NCE目标函数的梯度和MLE对数似然函数梯度是等价的, 也就是说我们通过NCE转换后的优化目标, 本质上就是对极大似然估计方法的一种近似, 并且随着负样本和正样本数量比的增大, 这种近似越精确, 这也解释了为什么作者建议我们将设置的越大越好。  
4)经验风险最小化在建模条件概率分布以及损失函数为对数情况下等价于极大似然估计。这个期望其实就是经验风险中样本数量趋于无穷大时等价于期望风险。
