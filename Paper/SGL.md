+ ***[Self-supervised Graph Learning for Recommendation(SGL)](https://dl.acm.org/doi/abs/10.1145/3404835.3462862)***   

![image-20220901140709165](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220901140709165.png)

**年份**：  2021

**引用次数**： 132

**应用领域**：  对比学习推荐系统,  基于CF

**方法及优缺点**：

作者认为[LightGCN](./LightGCN.md)和[PinSage](./PinSage.md)有两个限制, 高度数节点在特征学习上作用更强. 损害了低度数节点的特征学习, 其次, 特征容易受到噪声影响, 由于agg的方案会扩大观察到的边的影响. 所以为了加强鲁棒性和准确度增加了辅助的自监督任务, 设计了三种生成视角, **node dropout, edge dropout, random walk**, 除此以外作者还发现使用硬负样本会有其他作用, 不仅提升了模型表现还加速了训练过程.

**动机**:  

GCN只有稀疏的监督信号, 歪曲的数据分布, 以及交互的噪声等问题, 提出了SGL模型, 本模型是模型无关的可以使用其他的GNN模型, 只是在本论文中使用的是LightGCN模型.

**相关工作和理论**：  

使用的main函数就是pairwise的BPR loss, 公式如下所示.
$$
\mathcal L_{main} = \sum_{(u,i,j)\in O} - log\sigma(\hat{y}_{ui}-\hat{y}_uj)
$$
使用的数据增强如下:

**Node Dropout(ND)**:  $M', M''\in \{0,1\}^{|\mathcal V|}$是两个mask节点的向量去生成不同的子图
$$
s_1(\mathcal G) = (M'\odot \mathcal V, \mathcal E), s_2(\mathcal G) =(M''\odot \mathcal V, \mathcal E),
$$
**Edge Dropout(ED)**:$M_1, M_2\in \{0,1\}^{|\mathcal E|}$是两个mask边的向量去生成不同的子图
$$
s_1(\mathcal G) = (\mathcal V, M_1 \odot \mathcal E), s_2(\mathcal G) =(\mathcal V, M_2 \odot \mathcal E),
$$
**Random Walk(RW)**: 上面两个操作生成的子图在所有层共享, RW是在每一层有不同的mask边向量, 可以通过这种方式来模拟RW.
$$
s_1(\mathcal G) = (\mathcal V, M_1^{(l)} \odot \mathcal E), s_2(\mathcal G) =(\mathcal V, M_2^{(l)} \odot \mathcal E),
$$
接着是对比学习的内容, 使用的就是InfoNCE的公式 比较的就是加强后的两个视角下的相似度, $\mathcal L_{ssl}^{item}$同理, $\mathcal L_{ssl}=\mathcal L_{ssl}^{item}+\mathcal L_{ssl}^{user}$
$$
\mathcal L_{ssl}^{user} = \sum_{u\in\mathcal U}-log\frac{exp(s(z_u', z_u'')/\tau)}{\sum_{v \in \mathcal U}exp(s(z_u', z_v'')/\tau)}
$$
多任务训练的目标函数如下, 在$\mathcal L_{ssl}$中预训练, 在$\mathcal L_{main}$中微调.
$$
\mathcal L = \mathcal L_{main}+\lambda_1\mathcal L_{ssl}+\lambda_2||\Theta||^2_2
$$
后面作者理论分析了为什么硬负样本在SGL中有用的原因, 看不懂, 说直白点就是硬负样本提供更多的梯度帮助训练

大致理解如下, SSL Loss对向量的梯度结果如下, 其中$c(v)$代表了正负样本之间的关系

![image-20220901154715911](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220901154715911.png)

其中$c(v)$代表了和负样本的梯度, 这个值L2范数信息如下, 用相似度$x=s_u'^Ts_v''$代替, 如下所示

![image-20220901154935110](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220901154935110.png)

![image-20220901155019221](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220901155019221.png)

画出来$\tau = 1, 0.1$的图如下, 可以看出在0.1时, 相似度越高的负样本的梯度更加高, 有利于加速训练, 而相似度低的easy negative sample则几乎没有梯度.

![image-20220901155103578](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220901155103578.png)

为了研究$\tau$的大小, 对g(x)求导得到最值点的x为$x^*=\frac{\sqrt{\tau^2+4}-\tau}{2}$得到g(x)的最大值为$g(x^*)$, 然后取对数如下所示

![image-20220901160116021](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220901160116021.png)

$x^*$对$\tau$的图, $lng(x^*)$对$\tau$的图如下所示,随着𝜏的降低，影响最大的负节点与正节点变得更加相似(即$𝑥^∗$接近0.9)，而且它们的贡献被超指数放大(即𝑔∗接近𝑒8)。因此，正确设置𝜏使SGL能够自动执行硬负挖掘。

![image-20220901160235659](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220901160235659.png)



**实验结果**:  

复现回来补, SELFRec有

**个人总结**：  

图对比的经典文章, 说实话模型没什么创新的, 就是一个数据增强

**备注**  

