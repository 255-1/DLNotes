+ ***[Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding](https://dl.acm.org/doi/abs/10.1145/3159652.3159656)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206101606815.PNG)

**年份**：  2018

**引用次数**： 587

**应用领域**：  序列推荐, CNN

**方法及优缺点**：

Caser使用了CNN应用于NLP的思想提取序列的特征,既学习到序列模式的特征又捕获用户偏好, 具体来说序列模式能捕获到point-level和union-level两种, 用户偏好能捕获到skip behavior. 简单来说就是再对用户点击过的物品建模,得到embedding的基础上, 再加一个用户向量作为一般偏好一起训练得到结果.

**结论**：

同上.

**动机**:

 传统的Markovs的办法只能得到point-level,无法得到union-level序列模式,  而且不能得到skip behavior, 如下图所示, union-level的模式将前面的action整合在一起推荐, 比如购买硬盘和内存联合起来会给你推荐主板, 而单独每一项其实都不太会推荐主板.skip behavior就是过去的行为在跨过几个step后依然有用.

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206101538407.PNG)

作者为了证明union和skip有用, 计算了$\frac{sup(XY)}{sup(X)}$也就是对于输入$X$得到$Y$的置信度,  计算了不同长度的$X$,并且去除了出现频率<5和置信度<50%的 结果如下图, 可以看到在长度>2的情况下, 有许多union的存在和skip once, twice的存在.

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206101558852.PNG" style="zoom:50%;" />

Caser在此基础上应运而生, 通过使用横向CNN和纵向CNN捕获point-level, union-level和skip behavior, Caser 对用户的一般偏好和顺序模式进行建模, 并在单个统一框架中概括了几种现有的状态方法.

除此以外RNN也有它的局限性, 顺序推荐问题中，并非所有相邻的动作都有依赖关系（例如，一个用户在 i1 之后购买了 i2，只是因为她喜欢 i2）, 在数据具有很强烈的序列关系下RNN有优势,

**相关工作和理论**：  

作者分别介绍了模型的三个模块, Embedding查找, 卷积层 和全连接层

首先介绍Embedding查找,定义用户u在t时刻的矩阵如下, $E^{(u,t)}\in \mathbb R^{L\times d}$还定义了用户特征$P_u\in \mathbb R^d$
$$
E^{(u,t)}=\begin{bmatrix}Q_{S^u_{t-L}} \\
.\\
Q_{S^u_{t-2}}\\
Q_{S^u_{t-1}}
\end{bmatrix}
$$
接着介绍卷积层用来学习用户的短期兴趣, 将序列物品的矩阵看成是一张latent space的图, 做横向卷积和纵向卷积

对于横向卷积示意图如下, 用来获取union-level数据, 如图联合了飞机和酒店推荐了长城, 联合了快餐和餐厅推荐了酒吧

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206101652568.PNG" style="zoom:50%;" />

对于过滤器$F^k\in \mathbb R^{h\times d} 1\le k\le n. h\in\{1,..,L\}$ , n为过滤器数量, 比如对于L=4, 过滤器n=8, 则对不同的h=1, h=2, h=3, h=4分别分配2个过滤器, 对于不同h的过滤器卷积位置不一样, 所以其中一个的卷积公式如下, 使用$i+h-1$是为了保留最后的位置给大的h值.
$$
c_i^k  =\phi_c(E_{i:i+h-1 }\odot F^k)
$$
对于一个滤波器在整个"image"上的结果为concate在一起, 如下所示
$$
c^k = [c_1^k, c_2^k \cdot\cdot\cdot c^k_{L-h+1}]
$$
最后对不同$c^k$都使用max pooling得到最大值,所以输出的$o\in\mathbb R^n$公式如下, 行卷积每次都是对连续的h个物品进行卷积, 通过此来学习union-level的模式
$$
o = \{max(c^1), max(c^2), \cdot\cdot\cdot , max(c^n)\}
$$
接着作者介绍列卷积,  卷积核为$\tilde F^k\in\mathbb R^{L\times 1}$有$\tilde n$个过滤器, 由于一次一列, 所以并会出现卷积宽度变化, 所以公式如下
$$
\tilde c^k = [\tilde c^k_1\tilde c^k_2\cdot\cdot\cdot\tilde c^k_d]
$$
对列做卷积可以转换成对行做卷积, 结果一样, 所以上述的公式可以写成如下形式, 相当于做h=1的行卷积不过额外考虑了权重和, 这样的计算能学习到point-level的模式, 
$$
\tilde c^k=\sum_{l=1}^L\tilde F^k_l\cdot E_l
$$
最后将$\tilde n$全合并到一起得到$\tilde o\in\mathbb R^{d\tilde n}$, 公式如下
$$
\tilde o = [\tilde c^1 \tilde c^2\cdot\cdot\cdot\tilde c^\tilde n]
$$
在此作者总结了列卷积和行卷积的不同之处

1. 列卷积考虑的是序列之间的关系, 所以每次处理一列即可
2. 列卷积没有使用max pooling是为了能保留每个维度的结果



最后作者介绍全连接层的内容.将$o$和$\tilde o$级联在一起送到全连接层中提取high-level特征公式如下,得到的z称为卷积序列特征
$$
z = \phi_a(W\begin{bmatrix}o\\\tilde o\end{bmatrix}+b)
$$
将上述的卷积序列特征作为短期兴趣, 用户特征$P_u$作为长期兴趣合并在一起送入全连接层得到$y^{(u, t)}$, 代表用户u在时刻t对所有物品的点击概率, 公式如下,在这里作者之所以要把用户特征在这里级联,一方面是为了和其他模型统一,另一方面可以使用其他模型的参数与训练本模型. 
$$
y^{(u, t)} = W'\begin{bmatrix}z\\P_u\end{bmatrix}+b'
$$
最后作者介绍网络训练, 在loss function中完成skip behavior的方法, 对于输出层可以得到结果概率公式如下
$$
p(S^u_t|S^u_{t-1},S^u_{t-2},\cdot\cdot\cdot,S^u_{t-L}) = \sigma(y^{(u,t)}_{S^{u}_{t}})
$$
将序列分解为递增长度的集合$C^u={L+1, L+2, ....,|S^u|}$计算似然估计, 公式如下

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206101805340.PNG)

为了完成skip behavior将接下来推荐的物品定义为$D^u_t=\{S^u_t,S^u_{t+1},...,S^u_{t+T}\}$补充进上述的$C^u$中, 然后使用-log算法得到loss function如下所示, 对于$j\neq i$的负样本作者使用随机采样3个

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206101810363.PNG) 

**实验结果**:

复现回来补  

**个人总结**：  

CNN用于序列推荐的经典论文.

**备注**  