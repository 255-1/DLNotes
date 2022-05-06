+ ***[Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks(HRNN)](https://arxiv.org/abs/1706.04148)***  

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205061255428.PNG)

**年份**：2017  
**引用次数**：335  
**应用领域**：序列推荐  
**方法及优缺点**：

在session-based推荐系统中用户的行为可能很难获取, 但是也可能已经拥有一些可用的数据. 作者提出了一个无缝的个性化RNN模型可以处理session之间的信息转换, 并且设计了一个分层的RNN模型结构, 该模型可以在用户的session之间传递和演化隐藏状态.

**结论**：

本文提出的模型HRNN在之前RNN-based session之上额外使用了一个GRU建模了用户在不同session之间的行为和兴趣的演化. HRNN将获取的用户长期兴趣转移到了session层面. 这个效果在benchmark上效果很好.作者接着展望了添加注意力模型以及使用物品用户的特征来改善模型. 并且将personalized session-based模型用于其他领域.

**动机**:  

用户在一些系统中会是以登录的的或者有cookie来这种用户识别的形式. 在这些情况下可以假设用户之前的session可能提供有价值的信息给下一个session. 如果知识简单的concate过去和现在的session效果一般. 所以提出了本文的HRNN模型,解决了两个问题

1. session-aware推荐, 将用户过去的session的信息传播到下一个session
2. session-based推荐, 当没有过去的session, 该算法基于一个分层的RNN, 一个session结束时low-level的RNN的隐状态会传递到一个higher-level的RNN中为了预测一个好的context vector, 以便为下一个session开始的RNN隐状态提供良好的初始化.

**相关工作和理论**：  

首先是session层面的RNN, 和[GRU4Rec](./Session-Based_Recommendations_with_recurrent_neural_networks.md)一样, 使用的loss function为TOP1, 所使用的正负样本也和GRU4Rec一致,session层的公式为

$s_{m,n}=GRU_{ses}(i_{m,n},s_{m,n-1})$ 其中$s$为第$n$步的隐状态.

接着作者介绍HRNN模型, 主要完成了2点:  

1. 添加一个额外的GRU层去追踪用户的的兴趣转变.  
2. 使用user-parallel mini-batch提高训练效率.    

对于1的目标, 公式就是$c_m=GRU_{user}(s_m, c_{m-1})$每个session结束的时候把这个session的特征更新用户的特征.而这个用户特征将用来更新下一个session的初始化隐状态,  

$s_{m+1, 0} = tanh(W_{init}c_m+b_{init})$

这将user-level的信息又转换成了session-level的信息.进一步在下一个session-level 的更新如下图所示, 和之前的session-level的公式差不多, 其中括号的内容代表,用户特征是一个可选项目.

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205061316094.PNG)

$GRU_{user}$只会在session之间进行更新. 当上面的公式中选择传入的$c_m$时,即使$c_m$不变,也会在session中更新$GRU_{user}$

在HRNN中, 用户的推荐可能会因为之前session的点击而不同.

对于目标2是为了提高学习效率,类似GRU4Rec中的思想不过是基于不同user的, 示意图如下图所示,这种方法不仅能提高效率, 而且同batch下的负样本采样也是不同的其他用户的,就不会出现同一个用户的不同session的正样本变成负样本的污染情况, 采样方法依旧是基于流行度的采样办法.

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205061332159.PNG)

**实验结果**:  

复现回来补  

**个人总结**：  

引入了用户兴趣变化的特征, 使用额外的GRU学习, 不难, 但是这篇文章的写作并不避讳写出来某种方法效果不好的情况. 就直接说某种方法在这个模型中的效果不好, 所以并没有选择.

**备注** :  

