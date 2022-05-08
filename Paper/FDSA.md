+ ***[Feature-level Deeper Self-Attention Network for Sequential Recommendation(FDSA)](https://www.ijcai.org/proceedings/2019/600)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202205081137557.PNG)

**年份**：2019  
**引用次数**：55  
**应用领域**：序列推荐, 自注意力  
**方法及优缺点**：

现有的模型只考虑了序列中物品的转换模式, 忽视了物品的特征(种类, 品牌)之间的转换模式. 所以提出了本文的FDSA模型, 首次将不同的物品特征整合进特征序列中, 将推荐过程分成item-level和feature-level两个方向, 并且各自使用自注意力块进行学习.  最后整合到一起经过两层的全连接层进行物品推荐.

**结论**：

同上.  

**动机**: 

前人的FPMC, GRU4Rec, SASRec, 都只是考虑了物品之间的转换模式, 但是我们日常生活行为一般都是特征级别的, 但是很多物品的类别特征的占比人工很难识别到, 所以引入了自注意力层来辅助feature-level的工作, 此外, 物品类别一般是描述性文本, 所以需要从这种非结构化数据中提取出潜在特征.  
**相关工作和理论**：

作者从Embedding layer, vanilla attention layer, self-attention block和loss function分开介绍了FDSA的整体设计. 输入的物品从one-hot编码转换成低维的稠密向量, 比如对于物品描述可以使用topic model提取出关键字, 然后使用word2vec转换成向量,  对于物品类别这种特征 通过vanilla attention layer提取出重要的特征, 然后对item-level和feature-level的特征分别做自注意力网络训练, 将输出concate在一起经过一个全连接层得到最后的输出

首先是Embedding layer, 首先将序列转换成模型能处理的最大大小的$n$, 如果不够就在左边补零向量, 如果超过就截断, 和[SASRec](./Self-Attentive_Sequential_Recommendation.md)差不多. 对于对于序列的每个物品替换成为物品的种类信息$c=(c_1, c_2, ...,c_n)$通过一个lookup layer将类别的one-hot编码变成一个向量, 对于物品的文本描述使用topic model提取出关键字, 然后使用word2vec转换成向量.

接着介绍 vanilla attention layer, 这层的作用是为了把embedding layer中提取出来的特征找到其中重要的特征,对于$item_i$他的特征可以写成$A_i=\{vec(c_1), vec(b_1), vec(item^{text})\}$则attention公式如下, $f$为输出的特征
$$
\alpha_i=softmax(W^fA_i+b^f) \\
f_i = \alpha_iA_i
$$
然后介绍 self-attention block, 和[BERT4Rec](./BERT4Rec.md)中的Trm层类似,做LayerNorm和激活函数有点不同.

在上面得到了item-level和feature-level的输出$O_s^{(q)},O_f^{(q)}$后进过一个全连接层如下面的公式所示, $N$为物品的embedding矩阵. 输出的$y$就是在给定的前$t$个时刻的数据, 输出$item_i$的相关性.
$$
O_{sf}=[O_s^{(q)};O_f^{(q)}]W_{sf}+b_{sf}\\
y^u_{t,i} =O_{sf_t}N^T_i
$$
最后作者使用的是cross-entropy损失函数, 公式如下图所示
$$
L=-\sum_{i \in s}\sum_{t\in [1,2,...,n]}[log(\sigma(y_{t,i}))+\sum_{j\notin s}log(1-\sigma(y_{t,j})]
$$
**实验结果**:  

复现回来补.

**个人总结**：  
**备注**  