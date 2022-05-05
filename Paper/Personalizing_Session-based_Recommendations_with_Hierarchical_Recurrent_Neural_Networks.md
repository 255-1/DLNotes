+ ***[Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks(HRNN)](https://arxiv.org/abs/1706.04148)***  

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

**相关工作**：  
**实验结果**:  
**个人总结**：  
**备注** :  

