+ ***[Bootstrapping User and Item Representations for One-Class Collaborative Filtering](https://dl.acm.org/doi/abs/10.1145/3404835.3462935)***   

![image-20220831123643111](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220831123643111.png)

**年份**：  2021

**引用次数**： 23

**应用领域**：  对比学习推荐系统

**方法及优缺点**：

BPR的方法为了判别正样本和负样本, 前人工作更多的依赖于负样本采样, 但是这种情况下可能会让"未被观察的正样本"定义为负样本. 本文的BUIR提出了一种不需要负样本, 不仅让正样本之间的相关性更强, 也能防止模型塌陷.

BUIR有两个encoder, 第一个online encoder用来预测第二个encoder的输出, 第二个target encoder通过慢慢近似第一个encoder提供一个稳定的目标.BUIR通过直接最小化item和user的交叉预测误差来学习特征.  除此以外BUIR使用数据增强input来缓解数据稀疏问题.

**动机**:  

**相关工作和理论**：  

BUIR的loss有两个, 分别是online u-> target v和online v-> target u的和公式如下, 两个归一化向量的mse等价于内积的负数. 使用u->v而不死u->u因为会模型坍塌, 最后训练出来的参数是一样的, 传统的end-to-end不可行.

![image-20220831151531135](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220831151531135.png)

其中$\theta$是普通的梯度下降, $\xi$ 是来自$\theta$的动量更新, 公式如下, $\xi$的动量更新让target更加稳定这才让bootstrap有成效. 这种不同更新的办法不会让模型塌陷.

![image-20220831152017134](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220831152017134.png)

最后的交互分数公式如下, 只是用了online encoder变量$\theta$,之所以没有使用归一化的结果,因为作者发现归一化后的特征没能表现用户和物品的流行度(不懂为什么, 难道作者特征是通过相加得到的? 不然把流行度作为一个额外的属性计数不行吗)

![image-20220831153656916](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220831153656916.png)

接着作者介绍了他使用的数据增强的方法. Neighbour-based Data Augmentation. 对于输入的user和item, 分别找到user交互过的物品合集 和item被交互过的user合集作为增强的multi-hot inputs进行训练. 如下所示, 这种办法并不是增加了交互的数据量,而是使用了原始的数据,这让encoder可以学习到略带有扰动的信息, 因为并不是使用直接的正样本

![image-20220831170940348](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220831170940348.png)

数据增强的公式如下所示, 就是挑选了对应neighbor的子集,也就是把原本的两个one-hot 向量转成了两个multi-hot向量, 后面的学习过程都一样.

![image-20220831171740368](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220831171740368.png)

**实验结果**:  

复现回来补

**个人总结**：  

BYOL没看过, 所以对这里面的内容还有点不太习惯, 大致内容能看懂, 就是不要负样本, 左脚踩右脚上天呗.

**备注**  