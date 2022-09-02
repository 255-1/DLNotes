+ ***[Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation(SimGCL)](https://dl.acm.org/doi/abs/10.1145/3477495.3531937)***   

**年份**： 2021 

**引用次数**： 9

**应用领域**：  对比学习推荐系统, 基于CF

**方法及优缺点**：

作者想要知道为什么CL方法能得到更好表现效果, 得到在基于CL的推荐模型中，CL通过学习更均匀分布的用户/物品表示来操作，这可以隐式地减轻受欢迎程度偏差, 同时作者得到了图的增强并不是必须的, 反而是添加一些均匀的噪声到embedding里面会得到更好的效果. 最后作者得到在基于CL的模型中, CL loss是核心, 而图增强只是一些次要角色, 优化CL loss有助于在推荐场景下去偏见.

**动机**:  

作者首先设计了一个图增强的对比试验, 然后可视化了embedding的数据分布(t-SNE). 一方面优化InfoNCE得到更均等的特征分布, 而且无论有无图增强, InfoNCE对减缓流行度偏差有作用. 同时尽管图增强并不是很符合预期效果, 但是一样有作用, 适量的扰动帮助学习了特征. 而且edge dropout可能会让原本相连的图断开.

图对比学习示意图如下, 和[SGL](./SGL.md)中的内容差不多, 只不过这里给了个新的名词joint learning. 图特征学习用的就是[LightGCN](./LightGCN.md), 得到的结果就是没有所谓的数据增强, 只是使用对比学习也能取得很好的效果. 

![image-20220902104813631](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220902104813631.png)

学习到的特征的可视化图形如下所示, LightGCN的feature分布更加聚集, 而CL Only太平均了, 对于高聚集性分布作者认为有两个原因,  LightGCN使用的消息传递机制随着layer增加, 特征都趋近相似, 第二种原因是数据中的流行度偏见, 长尾物品很难训练.![image-20220902110843897](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220902110843897.png)

 CL Only太平均了, 因为将CL Loss重写为如下形式, 前面的$\tau$是常量, 所以只会缩小后面的相似度, 所以会越来越像. 但是这个过程可以看成是一个debias的过程. 综上希望能找到一个既没有LighGCN这样的bias的也不像CL Only这样太平均的.

![image-20220902112805712](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220902112805712.png)

![image-20220902112821926](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220902112821926.png)



**相关工作和理论**：  

**首先是对node$i$和它的特征$e_i$进行一个扰动**, 公式如下, 这个扰动对不同节点扰动都不同
$$
e_i'=e_i+\Delta_i', e''_i=e_i+\Delta_i'' \\
||\Delta||_2=\epsilon \\
\Delta=\bar{\Delta}\odot sign(e_i), \bar{\Delta}\in \mathbb R^d  \backsim U(0,1)
$$
可视化情况如下, 相当于将原来的向量转动了$\theta$的角度, 得到的两个新的向量就可以看成是一个augment的特征

![image-20220902114340618](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220902114340618.png)

其次是调节均匀性, 本模型使用cl loss的系数和上面扰动的$\epsilon$控制统一性, 更大的$\epsilon$会让最后的特征更均匀，因为当增强表示与原始表示足够远时，其表示中的信息也会受到噪声的相当大的影响, 对同一性有一个矩阵指标能表示,平均成对高斯势的对数(亦称为：径向基函数(RBF)核）, 公式如下, f()输出向量的L2范数,  这个Loss 越小越均匀

![image-20220902120023931](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220902120023931.png)

作者对SGL的变种和本文的SimGCL以对比了, 图片如下所示, 可以看出SimGCL比起SGL更加均匀.随着训练的进行，均匀度逐渐下降(loss越来越高)，达到峰值后，均匀度逐渐提高，直至收敛，并保持这一趋势. 而且SimGCL比起SGL最终更容易得到更均匀的结果. 而且也可以看出通过noise-based数据加强比起图增强会效果更好.

![image-20220902120509973](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220902120509973.png)

最后作者分析了时间复杂度, 需要再回来看.



**实验结果**:  

**个人总结**：  

蛮有意思的, 以前一直知道的是加点噪声能得到更好的训练效果, 这篇文章相当于说普通的加噪声和原图作对比, 比起其他有的没的增强办法都要好.

**备注**  

Understanding contrastive representation learning through alignment and uniformity on the hypersphere.