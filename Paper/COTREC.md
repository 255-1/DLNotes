+ ***[Self-Supervised Graph Co-Training for Session-based Recommendation](https://dl.acm.org/doi/abs/10.1145/3459637.3482388)***   

**年份**：  2021

**引用次数**： 22

**应用领域**：  对比学习序列推荐

**方法及优缺点**：

session-based序列推荐比起其他的推荐方法更加会受到数据稀疏问题的影响. 现在常用的对比学习的数据增强方法比如item/segment dropout会让session-based更加稀疏. 本文中作者将自监督学习和联合学习合并在一起. 首先将一个序列增强成双视角(内部联系和外部联系).它们递归地利用不同的联系来生成真实样本，通过对比学习来监督彼此

**动机**:  

co-training是一个半监督学习的办法, 基本思想是训练两个分类器处理两个views接着预测unlabeled实例的伪标签, 以迭代方式互相监督.

**相关工作和理论**：  

**实验结果**:  

**个人总结**：  

**备注**  