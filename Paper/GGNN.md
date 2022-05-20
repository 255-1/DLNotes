+ ***[GATED GRAPH SEQUENCE NEURAL NETWORKS(GGNN)](https://arxiv.org/abs/1511.05493)***   

**年份**： 2015

**引用次数**： 1860

**应用领域**：GNN, Sequential Data  

**方法及优缺点**：

将GRU拓宽到GNN中间, 让GNN能够处理序列数据得到对应的feature.

**结论**：

**动机**:  

本模型主要弥补了GNN一般被用来做分类任务, 而不能输出序列的问题, 比如路径预测. 图上的特征学习分为两步, 第一从输入中学习representation, 加入了一些小改动用来适配RNN,第二在一连串输出中学习到隐变量的representation, 也是本文的重点.

**相关工作和理论**：  

作者首先介绍了GNN的做法, 从传播和输出两个步骤, 主要参考了2009的GNN开山作.

对于GNN的传播过程,公式如下,使用递归的办法直到收敛就得到了特征$h_v^{(t)}$这其中涉及到了当前节点的label, 和当前节点出入连接的边label, 周围邻居的label, 和邻居前一时刻$t-1$的特征这些变量.

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205201704550.PNG)

将上面的公式的$f^*$分解一下, 得到如下公式,本质没有变, 分解成了两个$f$函数

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205201708095.PNG)

$f$函数可以通过一个linear transform用nn学习, 对于入度的情况,公式如下

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205201715917.PNG)

接着作者介绍GNN的输出模型和学习, 输出模型主要是学习$g(\bold h_v, l_v)$, GNN中专注于收敛时候的每个节点的独立特征$h_v^{(T)}$, 对于readout的获取(graph-level)可以创造一个不存在的超节点通过特殊边和所有节点相连. 对于模型的学习通过Almeida-Pineda algorithm(备注1) ,这种办法迭代固定点来讲隐藏状态收敛,但是对初始条件要求高, 需要是压缩映射.

接着作者提出了本文的重点GGNN, 将GRU用于上述的GNN中, 将之前学习的时间步固定为$T$,而不是一直学到收敛,使用BPTT(backpropagation through time)学习梯度,

首先是node annotation, 个人理解就是节点基于任务会有个特殊标记, 比如可达性任务中起始节点标记[1,0],结束标记为[0,1],其余节点标记为[0,0], 直接用这个标记初始化对应节点的特征$h$, 扩展维度就补0. 这样是为了输出时开头的维度就能得到结果.

接着是GGNN的传播模型介绍,公式如下,(1)就是上面的node annotation补0的特征, (3)-(6)就是GRU的公式, 重点是Eq(2)

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205201758310.PNG)

$\bold A$由出度和入度矩阵拼接, 如下图的(c), Eq(2)可以看成是通过当前节点的出度和入度的值和对应顶点的特征做加权和,以此来学习传递的信息, 总的来讲就是Eq(2)汇聚了周围的信息, 后面的GRU-like的公式用来决定如何更新下一层layer的节点特征.

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205201800079.PNG" style="zoom:60%;" />

接着就是介绍GGNN的输出模型, 和GNN一样都可以使用$g(\bold h_v, l_v)$学习一个输出,其次在GGNN中的readout公式如下,$\sigma(\cdot)$为soft attention机制, 判断哪些节点和readout有关, 其中$i$和$j$都是一个nn, 输入是$concate(h_v^{(T)},x_v)$输出是一个实数

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205201813325.PNG)

GGNN的输出只是一个结果, 如果需要输出一个序列需要本文的GGSNN做序列输出, 做法是使用了两个GGNN分别用来预测output和下一个隐状态, 示意图如下, 有需要再回来研究.

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205201827571.PNG)

**实验结果**:  

复现回来补

**个人总结**：  

个人感觉主要内容就到GGNN, 其中的这个A矩阵用法的设计很厉害, 能很好的处理有向图.但总体来说他的本质和其他GNN一样, 都是汇聚信息然后更新.

**备注**  

[Almeida–Pineda algorithm](http://blog.nodetopo.com/2019/12/03/almeida-pineda-algorithm/)