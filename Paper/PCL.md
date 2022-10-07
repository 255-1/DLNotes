+ ***[Prototypical Contrastive Learning Of  Unsupervised Representation(PCL)](https://arxiv.org/abs/2005.04966)***   

![image-20221007143407864](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20221007143407864.png)

**年份**：  2020

**引用次数**： 342

**应用领域**：  无监督对比学习, 聚类算法

**方法及优缺点**：

PCL在个体判别的基础上引入聚类算法, 这样不仅可以学习到low-level的特征区别还可以学习到隐语义结构信息. 对于聚类算法使用EM算法来极大似然估计来优化网络参数$\theta$ 

**动机**:  

个体判别任务有个很大的问题就是每个instance作为一个positive pair和其他的instance都是negative pair, 即使你们相似也不会管, 所以没有考虑语义信息. 所以本文使用聚类算法让每个instance和自己的聚类中心更加靠近.

作者将PCL作为一个EM算法去求优化网络参数, 而且本文推出了ProtoNCE属于是InfoNCE的推广情况. 可以评估不同聚类中心的特征聚集情况.

**相关工作和理论**：  

一个模型的参数的MLE公式如下
$$
\theta^*=\arg\max\limits_{\theta}\sum_{i=1}^{n}\log p(x_i|\theta)
$$
不妨**假设data$\{x_i\}_{i=1}^n$和隐藏变量相关**$C={c_i}_{i=1}^k$, 由此重写公式(1)的内容
$$
\theta^*=\arg\max\limits_{\theta}\sum_{i=1}^{n}\log p(x_i|\theta)=\arg\max\limits_{\theta}\sum_{i=1}^{n}\log \sum_{c_i \in C} p(x_i, c_i|\theta)
$$
但是这个公式没有办法直接计算因为log里面是累加,相当于是连乘, 所以使用了琴升不等式得到这个目标函数的下界
$$
\sum_{i=1}^{n}\log \sum_{c_i \in C} p(x_i, c_i|\theta)=\sum_{i=1}^{n}\log \sum_{c_i \in C} Q(c_i) \frac{p(x_i, c_i|\theta)}{Q(c_i)}\ge \sum_{i=1}^{n}\sum_{c_i \in C} Q(c_i) \log \frac{p(x_i, c_i|\theta)}{Q(c_i)}
$$
为了符合琴升不等式的限制条件, 所以对于系数$\sum_{c_i \in C}Q(ci) = 1$加权和为1, 等号成立当且仅当$\frac{p(x_i, c_i|\theta)}{Q(c_i)}$ 为常数, 由如下公式可以看出$Q(c_i)$的概率是对$x_i$属于聚类中心$c$的概率密度函数
$$
\frac{p(x_i, c_i|\theta)}{Q(c_i)} = m \\
Q(c_i) = \frac{p(x_i, c_i|\theta)}{m} \\
m\sum_{c_i}Q(c_i) = \sum_{c_i}{p(x_i, c_i|\theta)} \\
m = p(x_i|\theta) \\
Q(c_i)=\frac{p(x_i, c_i|\theta)}{p(x_i|\theta)}=\frac{p(x_i|\theta)p(c_i|x_i,\theta)}{p(x_i|\theta)} = p(c_i|x_i,\theta)
$$
除此以外再舍去了和$x_i$无关的的分母的常量, 最后的优化目标就如下
$$
\sum_{i=1}^{n}\sum_{c_i \in C} Q(c_i) \log{p(x_i, c_i|\theta)}
$$

### E-step:

通过聚类得到$Q(c_i)= p(c_i|x_i,\theta)$ ,使用k-mean对特征空间$v_i=f_\theta(x_i)$进行聚类, 对于$x_i\in c_i$就为1, 否则为0,  这里的encoder使用Moco中的动量编码器来完成

### M-step:

在E-step的基础上使用我们的目标变成了如下
$$
\begin{equation*}
  \begin{aligned}
    \sum_{i=1}^{n}\sum_{c_i \in C} Q(c_i) \log{p(x_i, c_i|\theta)}=\sum_{i=1}^{n}\sum_{c_i \in C} p(c_i|x_i,\theta) \log{p(x_i, c_i|\theta)} \\
=\sum_{i=1}^{n}\sum_{c_i \in C} \mathbb 1(x_i\in c_i) \log{p(x_i, c_i|\theta)}
  \end{aligned}
\end{equation*}
$$
**在簇中心选取概率相等的假设下**
$$
p(x_i, c_i|\theta) = p(x_i|c_i, \theta)p(c_i,\theta)=\frac{1}{k}\cdot p(x_i|c_i, \theta)
$$
除此以外再**假设再每个簇下的sample分布为各向同性的高斯分布**, 每个维度之间也是互相独立的, 则$p(x_i|c_i, \theta)$公式如下, 其中$v_i=f_\theta(x_i)$
$$
p(x_i|c_i, \theta) = \exp (\frac{-(v_i-c_s)^2}{2\sigma_s^2})/\sum_{j=1}^k\exp (\frac{-(v_i-c_j)^2}{2\sigma_j^2})
$$
再假定$c, v_i$都是正则化后的向量, 则$(v-c)^2=2-2v\cdot c$, 则整个目标公式如下所示, 其中$\phi$是一个关注簇周围的关注度的指标, $\phi \propto \sigma_s^2$
$$
\theta^*=\arg \min\limits_{\theta}\sum_{i=1}^n-\log \frac{exp(v_i\cdot c_s/\phi_s)}{\sum_{j=1}^kexp(v_i\cdot c_j/\phi_j)}
$$
在实际中,会cluster M次, 每次都是不同的cluster 数量, 并且InfoNCE来保持局部平滑性, 由此得到ProtoNCE如下
$$
\mathcal L_{ProtoNCE}=\sum_{i=1}^n-(\log \frac{exp(v_i\cdot v_i'/\tau)}{\sum_{j=1}^kexp(v_i\cdot v_j'/\tau)} +\frac{1}{M}\sum_{i=1}^M\log \frac{exp(v_i\cdot c_s^m/\phi_s^m)}{\sum_{j=1}^kexp(v_i\cdot c_j^m/\phi_j^m)})
$$

### 簇集中度$\phi$

不同的簇中心周围汇聚的特征的稠密程度是不一样的. $\phi$越小, 集中度就越高, 可以看到分母在平均的基础上加上了log函数, 因为$\phi$设计在两种情况下比较小

1. 特征和簇中心相似
2. 在簇中心周围由许多数量的特征

$$
\phi = \frac{\sum_{z=1}^Z||v_z'-c||_2}{Z\log(Z+\alpha)}
$$

**实验结果**:  

**个人总结**：  

这个和Supervised Contrastive Learning感觉很像, SupCon主要是引入了标签信息, 让同一个标签内的尽量靠近, 但是没有让一个标签内的数据相互正交,  本文PCL直接在聚类的基础上使用了个体判别来做局部平滑性 .其次SupCon的计算主要是同类别下的相互的相似度, 而本文的PCL使用的是统一的聚类中心, 谁好谁坏还不太清楚后面需要可以回来看看,

**备注**  