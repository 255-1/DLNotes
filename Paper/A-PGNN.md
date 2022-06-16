+ ***[Personalized Graph Neural Networks with Attention Mechanism for Session-Aware Recommendation](https://ieeexplore.ieee.org/document/9226110)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206161534610.PNG)

**年份**：2019  

**引用次数**： 14

**应用领域**：序列推荐, GNN

**方法及优缺点**：

对于session-aware推荐本模型提出了两个主要组件. 第一, PGNN对用户的序列图额外考虑了用户特征. 第二使用Trm融合了历史session对当前session的影响.

**动机**:  

基于RNN的模型和[SRGNN](./SRGNN)这些session-based只是用了一个匿名session. 对于另一些HRGNN这些personalized model的模型没使用序列图来捕获complex item transition关系, 另一方面一些session-aware模型无法明确分辨不同的历史session的权重

**相关工作和理论**：  

变量注释如下图, 作者从几个方面介绍, **序列图生成, PGNN, 历史session融合, 用户特征生成和推荐预测**

<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206161553690.PNG" style="zoom:50%;" />

首先是**序列图生成**, 公式如下, 本质和[SRGNN](./SRGNN)中的归一化出度入度矩阵一样,只是是从矩阵元素角度写出的公式
$$
w^{out}_{ij}=\frac{Count(v_i, v_j)}{\sum_{v_i\to v_k}Count(v_i, v_k)} \\
w^{in}_{ij}=\frac{Count(v_i, v_j)}{\sum_{v_i\gets v_k}Count(v_k, v_i)} \\
A_u^{out}[i, j]=w^{out}_{ij}\\
A_u^{in}[i, j]=w^{in}_{ij}
$$
接着介绍**PGNN**, 公式如下, 本质和[SRGNN](./SRGNN)中的更新办法一致, 只是从矩阵元素角度写的公式, 并且加了一个额外的用户特征$e_u$, GNN的Update方式也和[SRGNN](./SRGNN)一致, 使用GRU.
$$
a^{(t)}_{out_i}=\sum_{v_i\to v_j}A_u^{out}[i, j][h_j^{(t-1)}||e_u]W_{out}\\
a^{(t)}_{in_i}=\sum_{v_j\to v_i}A_u^{in}[i, j][h_j^{(t-1)}||e_u]W_{in}\\
a_i^{(t)}=a^{(t)}_{out_i}||a^{(t)}_{in_i}
$$
然后介绍 **历史session融合**, 一个session的embedding就是其中物品的max-pooling的结果,用$f_i^u \in \mathbb R^d$表示,$F^u$代表用户过去session的特征
$$
f_{i,j}^u=max_{1\leq j\leq d}(h_{1,j},h_{2,j},...,h_{m_i,j})\\
F^u = [f_1^u,...,f_{n-1}^u]
$$
将当前session的并不使用max-pooling而是直接使用item embedding集合, 用$H^u=[h_1,h_2,...,h_m]$表示, 将$H^u$作为Trm中的Query, $F^u$作为Key和Value得到如下的Trm公式,由此得到了历史session对当前session的权重
$$
Q^u = Relu(H^uW^Q)\\
K^u = Relu(F^uW^K)\\
V^u = Relu(F^uW^V)\\
H_h = Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt {d_k}})V
$$
用户的总session特征就为$H^{u'}=H_h+H^u=[h_1',...,h_m']$

然后介绍**用户特征生成**, 参考了[SRGNN](./SRGNN)中的self-attention机制, 将最后一个物品$h_m'$作为local意图, 所有$h_i'$所谓global意图, 然后将local和global级联就是用户的动态意图了$z_d = z_g||z_l$ 接着将用户的动态意图和用户的长期意图$e_u$级联并通过一个线性变换B将维度映射回d, 公式如下
$$
z_u = B[z_d||e_u]
$$
最后是**推荐预测**, 计算分数就是将$z_u$和所有物品embedding计算内积, 然后通过softmax得到概率, 最后使用cross entropy, 涉及的公式如下
$$
\hat{z_i}=z_u^Te_{v_i}\\
\hat y = softmax(\hat{z_i})\\
\mathcal L(\hat y)=-\sum_{i=1}^{|V|}y_ilog(\hat{y_i})+(1-y_i)log(1-\hat {y_i})
$$
**实验结果**:  

复现回来补

**个人总结**：  

**备注**  