+ ***[Session-based Recommendation with Graph Neural Networks(SR-GNN)](https://arxiv.org/abs/1811.00855)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205181156199.PNG)

**年份**：2018

**引用次数**： 452

**应用领域**：  GNN, 序列推荐, 注意力机制

**方法及优缺点**：

之前的session-based推荐方法会建模一个序列并且也有考虑user representation, 但是这个user representation不够准确并且忽视了物品的复杂过度情况, 为了获取更加精准的嵌入我们提出了本文的SR-GNN模型. 将session的序列编码成图结构的形式, 变成序列图, 这样可以得到物品的复杂转换. 每个session都会用注意力机制编码成全局和当前兴趣的整合.

**结论**：

同上.

**动机**:

前人的MDP, [GRU4Rec](./Session-Based_Recommendations_with_recurrent_neural_networks.md), [NARM](./Neural_Attentive_Session-based_Recommendation.md)虽然效果都不错, 但是对于user representation都很难准确估计, 一般都会把RNN的隐状态当作user representation, 除此以外这些方法总是建模连续item之间的单向转换，而忽略上下文之间的转换, 为此引入了GNN.

本模型在将序列转变为序列图后, 使用GGNN训练每个node, 训练完成后将session表示为node的注意力处理结果.然后用session的representation去预测下一次点击.

**相关工作和理论**：  

作者介绍的主要流程是,生成序列图,节点学习, 生成session特征和预测四个部分

首先介绍生成序列图,序列关系变成有向图中的指向关系, 对于多个出度或者入读的节点就做归一化, 让权重和为1, 具体如下图所示.这个矩阵为$A_s$

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205181206562.PNG" style="zoom:67%;" />

接着介绍节点学习过程, 和GGNN类似,公式如下, 2-5就是GRU不谈了, 对于Eq1的个人理解是$A_{s,i:}$是上面拼接矩阵的一行, 包含当前节点的出度和入度,那对于session中点击的物品$v_i^{t-1}$就2次分别乘以出度和入度矩阵, 得到结果分别为归一化度的特征, 但是这里我有个很疑惑, $A_{s,i}$比方为出度矩阵[1Xn],那序列也必须要n长度, 没看代码不知道作者怎么处理补0和截断的. 对于$H$就是让特征升维度到2d

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205181208667.PNG" style="zoom:67%;" />

 然后在训练完的基础上生成session特征, 将最后一次点击作为local特征, 使用attention机制得到点击物品的评分,然后综合成session的global特征, 公式如下

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205181232322.PNG" style="zoom:67%;" />

然后将global和local的结果concate就是session的特征, $s_h=W_3[s_l;s_g]$

最后就是做预测,将$s_h$和每个物品做内积相似度,得到的结果softmax以下,使用交叉熵作为loss, 使用BPTT梯度回传, 很平常的做法
$$
\hat{\bold z_i}=s_h^T\bold v_i\\
\hat y= softmax(\hat{\bold z})\\
\mathcal L(\hat y)=-\sum_{i=1}^m\bold y_i\log(\hat{\bold y_i})+(1-\bold y_i)\log(1-\hat{\bold{y_i}})
$$
**实验结果**:  

复现回来补

**个人总结**：  

**备注**  