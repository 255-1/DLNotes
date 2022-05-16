+ ***[SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://arxiv.org/abs/1609.02907)***   

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205161141347.PNG" style="zoom:67%;" />

**年份**：  2016

**引用次数**： 10791

**应用领域**：  GNN

**方法及优缺点**： 

提出了一种在图结构上进行半监督学习的方法, 它使用切比雪夫的一阶近似估算, 模型复杂度和边数成线性, 并学习了编码局部图形结构和节点特征的隐藏层表示.

**结论**：

同上

**动机**: 

图节点的分类问题是一个半监督学习的问题, 一般只知道一部分节点的label去预测其他未知的节点, 有一种办法是假设 相邻的节点有相同的标签的办法, 公式如下,前面的$\mathcal{L}_0$为有label的节点的损失, 后面的正则化项为未归一化的拉普拉斯, 
$$
\mathcal{L}=\mathcal{L}_0+\lambda\mathcal{L}_{reg} \\
\mathcal{L}_{reg}=\sum_{i,j}A_{ij}||f(X_i)-f(X_j)||^2=f(X)^T\Delta f(X)
$$
 本文在这个基础上使用神经网络对图的结构进行编码, 避免使用正则化项, 相反通过设置f()的条件使得分开来有label数据的梯度和无label数据的梯度. 作者在这个和之前的[GCN](./GCN.md)的区别是直接引用在大规模数据上, 所以需要近似计算.

**相关工作和理论**：  

本文完成的两个目标:

1. 一个好的layer-wise传播规则, 并且使用一阶近似计算.
2. 提高效率.

首先作者直接介绍了layer-wise的传播公式如下,乍一看有点迷.需要介绍一点额外内容.
$$
H^{(l+1)}=\sigma(\tilde D^{-\frac{1}{2}}\tilde A\tilde D^{-\frac{1}{2}}H^{(l)}W^{(l)})
$$
对于图的卷积来说无论是从spatial角度(消息)还是spcetral角度(矩阵)都是为了聚合节点周围的信息,假设我的节点信息来自周围的节点的和, 如下图所示,那么可以写出公式$aggregate(X_i)=AX$ 

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205161038567.PNG" style="zoom:33%;" />

可以进一步思考上面这种聚合方法是没有考虑自身的, 所以我们加上自环$aggregate(X_i)=(A+I)X=AX+X$所以一种简化的写法$A+I$就出现了, 那还有一种也是最常用的一种聚合方式, 就是将当前节点和周围节点的差值求和, 也就是[GCN](./GCN.md)中拉普拉斯矩阵的定义$L=D-A$ 公式如下所示,

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205161043246.PNG" style="zoom:50%;" />

但是上面这些方法都有个很大的问题就是没有归一化,只是单纯的求和,比如上面添加自环的办法,写成$\tilde A=A+I$, 希望在自环的基础上能够归一化. 公式如下所示

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205161046026.PNG" style="zoom:33%;" />

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205161046178.PNG" style="zoom:50%;" />

这考虑了附近节点的权重, 那其实可以更近一步计算归一化, 将当前节点的度数和周围节点的度数也添加进来, 简单点来说就是两个点虽然直接相连但是其中一个节点和太多太多节点相连这种情况下其实它对另一个节点的作用并不大,可以使用几何平均数$\sqrt{\tilde D_{ii} \tilde D_{jj}}$,叫做对称归一化,公式如下

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205161051183.PNG" style="zoom:33%;" />

以上的这些不同的信息聚集的办法, 都可以统称为拉普拉斯变换, 都是一些变种, 这些变种是比较常用的.

本文的所使用的Eq2的公式就可以很简单的发现就是加了自环与对称归一化后的公式. 这其实也就是一种节点信息聚集的办法, 现在看Eq2就会发现它很好理解, 就是将上一层的feature做一个信息聚集然后做一个transform通过一个激活函数就好了.

本文就是在[GCN](./GCN.md)中使用的切比雪夫多项式使用K=1进行近似, 所以这种计算结果是线性, 使用这种近似的办法可以通过堆叠多个卷积核来完成特征提取, 除此以外这种近似可以缓解局部结构的过拟合问题, 并且由于是一阶近似所以可以让网络更深一点来达到更好的效果,

在这种近似下拉普拉斯分解的$\lambda_{max}\approx2$ 卷积的公式如下, 作者在这片文章提出的拉普拉斯变换为$L=I_N-D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$ 根据之前的解释, 可以理解成当前点的值减去周边节点对称归一化的结果.
$$
g_{\theta'}\star x\approx \theta'_0x+\theta'_1(L-I_N)x=\theta'_0x-\theta'_1D^{-\frac{1}{2}}AD^{-\frac{1}{2}}x
$$
在实际操作中, 可以进一步降低变量的数量来防止过拟合和训练复杂度, 公式如下, 把上面两个变量都整合在一起, 其中$\theta=\theta'_0=-\theta_1'$
$$
g_{\theta'}\star x\approx \theta(I_N+D^{-\frac{1}{2}}AD^{-\frac{1}{2}})x
$$
因为用的对称归一化操作后加上$I_N$的范围为[0,2], 希望能转到[0,1]防止梯度消失/爆炸, 所以这边用了个renormalization trick, $I_N+D^{-\frac{1}{2}}AD^{-\frac{1}{2}}\to \tilde D^{-\frac{1}{2}}\tilde A\tilde D^{-\frac{1}{2}}$ 其中$\tilde A=A+I$, $\tilde D$为$\tilde A$的度数矩阵.

综上提取出的feature公式如下,和Eq2的本质一样, 之所以要把权重$\Theta$放在后面是为了让输入的dim C能映射到更高维F中, 可以从最上面的架构图看出.
$$
Z  =\tilde D^{-\frac{1}{2}}\tilde A\tilde D^{-\frac{1}{2}}X\Theta
$$
作者后面举了个例子, 设计一个2层的GCN得到的特征如下,
$$
Z=softmax(\hat A ReLU(\hat AXW^{(0)})W^{(1)})
$$
最所有的labeled data使用交叉熵损失函数, 这里的F就是输出的特征的dim, 还没看代码,个人感觉这里的f没什么意义, 一般输出就是一个label, 不过也不一定, 说不定所谓的label就是一个vector.
$$
\mathcal L=-\sum_{l\in \mathcal Y_L}\sum_{f=1}^FY_{lf}lnZ_{lf}
$$
**实验结果**:  

复现回来补

**个人总结**：  

总体使用的是之前GCN中Cheb的一阶近似计算. 论文引用数量很大.

**备注**  