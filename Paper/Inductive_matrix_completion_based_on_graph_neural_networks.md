+ ***[Inductive matrix completion based on graph neural networks](https://arxiv.org/abs/1904.12058)***   
  
**年份**：2020  
**引用次数**：49  
**应用领域**：推荐系统  
**方法及优缺点**：
  选择这篇文章主要因为排名高效果好，并且有Pytorch源码研究，  

**结论**：
  user-item矩阵的补全传统需要使用side information做transductive补全，所学习的embedding结果无法推广到未见过的user，item甚至新的矩阵。传统的inductive补全需要高质量的内容side information，这很难获得。本论文模型IGMC使用GNN得到不用任何side information基础上得到inductive补全矩阵，核心在于IGMC纯粹基于从评级矩阵中产生的（用户，项目）对周围的1跳子图来训练GNN，并将这些子图映射到其相应的评级。

**动机**:  
**相关工作**：  
**实验结果**:  
**个人总结**：  
**备注**  
1.inductive learning和transductive learning的区别
> Inductive learning，翻译成中文可以叫做“归纳式学习”，顾名思义，就是从已有数据中归纳出模式来，应用于新的数据和任务。我们常用的机器学习模式，就是这样的∶根据已有数据，学习分类器，然后应用于新的数据或任务。  
Transductive learning，翻译成中文可以叫做“直推式字习”，指的是由当前学习的知识直接推广到给定的数据上。其实相当于是给了一些测试数据的情况下，集合已有的训练数据，看能不能推广到测试数据上。  
对应当下流行的学习任务:
> 1.Inductive learning对应于meta-learning(元学习)，要从诸多给定的任务和数据中学习通用的模式，迁移到未知的任务和数据上。  
> 2.Transductive learning对应于domain adaptation (领域自适应)，给定训练的数据包含了目标域数据，要求训练一个对目标域数据有最小误差的模型。
