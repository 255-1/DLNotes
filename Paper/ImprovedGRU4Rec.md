+ ***[Improved Recurrent Neural Networks for Session-based Recommendations](https://dl.acm.org/doi/abs/10.1145/2988450.2988452)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206111439230.PNG)

**年份**：  2016

**引用次数**： 431

**应用领域**：  序列推荐, RNN

**方法及优缺点**：

使用了NLP和CV领域中的方法改善[GRU4Rec](./Session-Based_Recommendations_with_recurrent_neural_networks.md),使用数据增强以及一种考虑了输入数据变换的方法, 并且作者借用了CV中的teacher-student用于预训练和微调, 最后作者并不是预测one-hot编码的物品, 而是直接预测下一个物品的embedding. 得到了很好的效果

**结论**：

同上

**动机**:  

前人的session-based推荐系统能弥补cold start问题, 本文就是在[GRU4Rec](./Session-Based_Recommendations_with_recurrent_neural_networks.md)的基础上进一步改进,主要完成了以下三点

1. 对序列预处理进行数据增强, 并且在embedding层加入dropout防止过拟合
2. 预训练模型以考虑了数据分布的时间转移
3. 对小数据集使用先验信息进行知识蒸馏

**相关工作和理论**：  

首先介绍的是数据增强部分, 示意图如下, 很直观, 将一个序列改为子序列的集合,并且使用dropout去除随机的节点,

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206111451167.PNG" style="zoom:50%;" />

接着作者介绍时间转移的问题, 机器学习有一个关键假设就是不同样本之间相互独立并且数据分布相同, 但是这件事情在推荐系统中不一定, 因为用户的兴趣是会随着时间变换的, 而且推荐给用的物品不会是用户点击过的而是一个新的物品, 所以学习完整的数据效果并不好, 模型会关注那些过时的属性, 作者的解决办法是在所有数据上预训练模型, 将预训练模型的参数初始化一个新模型, 这个新模型的只关注较新的数据集, 这就比较类似CV的在ImageNet上预训练在目标邻域的图片再微调的操作.

然后作者介绍使用先验信息进行知识蒸馏的方法. 物品点击序列在预测点击物品之后的物品可能也包含一些信息(上图中的蓝色物品), 但是这种信息在预测的时候是无法观察到的, 所以这些蓝色物品就是先验信息, 可以在训练的时候提供soft label用于正则化模型, 作者在这里使用了通用的知识蒸馏框架完成这件事.

在预训练的时候作者并不需要进行数据增强中的操作, 将原来的完整长度为$n$的序列转换为先验序列, 完整的序列为$[x_1, x_2,...,x_n]$通过子序列预处理得到$[x_1, x_2, ...,x_r] r\lt n$ ,则先验序列为$x^*=[x_n, x_{n-1},...,x_{r+2}]$ 就是将完整序列反转在后面添加下一次点击之后的物品,都是预测$x_{r+1}$ 在teacher模型中就按照先验序列训练. 之后微调student模型,loss function为$(1-\lambda)*L(M(x),V(x_n))+\lambda * L(M(x), M^*(x^*))$模型M即可以学习到real labels, 也可以学习到来自teacher model的$M^*$的结果. 这种办法对小数据集的非常有用. 

最后作者改善了输出层, 输出层如果为所有待选物品则参数量为$H\times N$其中$N$一般非常大, 在NLP中一般的做法是使用分层的softmax, 在这里不适用因为推荐数量大于1, 还有一种做法是抽样出高频率的物品. 作者在这里选择了直接预测下一个物品的embedding , 使用cosin计算预测的embedding和正确物品的embedding的相似度. 为的是让一个点击序列中的物品在embedding空间中尽量靠近,并且这样让最后的输出层的参数量为$H\times D$. 这种办法对与item embedding的质量要求高, 获得这种embedding的一种方法是从上述模型中提取并重用经过训练的项目嵌入

**实验结果**:  

复现回来补

**个人总结**：

本文不仅展示了NLP和RS的紧密关系, 而且我觉得很好展示了如何思考将CV和NLP邻域的发现应用于RS中.  

**备注**  