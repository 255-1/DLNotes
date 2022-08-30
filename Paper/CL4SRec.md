+ ***[Contrastive Learning for Sequential Recommendation(CL4SRec)](https://ieeexplore.ieee.org/abstract/document/9835621)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220830113005478.png)

**年份**： 2020

**引用次数**： 41

**应用领域**：  对比学习推荐系统

**方法及优缺点**：

普通的序列推荐由于数据的稀疏性导致很难学习到高质量的用户特征. 所以引入了对比学习框架去捕获自监督信号. 本论文使用了三种数据增强的方法(crop/mask/reorder)去生成自监督信号.

**结论**：

同上

**动机**:

传统序列模型比如GPT并不适用于推荐系统, 因为没有大量的语料库进行预训练, 而且不像NLP, 推荐系统的不同任务并不共享知识. 其次, GPT的自监督信号的目标函数和序列推荐的目标函数一样, 使用一样的目标函数并不能学习到更多有用的用户特征. 前人的工作着重于加强物品的表示, 本文提出的CL4Rec着重于用户行为序列的表示. 结合了传统序列推荐的目标和对比学习的目标函数, 让同一用户行为序列的不同增强视角尽可能相近来学习, 使用了三种数据增强的方法(crop/mask/reorder).

**相关工作和理论**：  

被SIMCLR启发, 对比学习框架分为三个部分, **1) 数据加强 2)用户特征编码 3)对比损失函数**, 其中用户特征学习使用的Transformer来学习序列$s_u$的特征. 其中SIMCLR会使用一个projection层, 但是在这里效果不好所以去除了. 对比损失函数如下,和以前学的一样.
$$
\mathcal L_{cl}(s_u^{a_i},s_u^{a_j}) = -log\frac{exp(sim(s_u^{a_i},s_u^{a_j}))}{exp(sim(s_u^{a_i},s_u^{a_j}))+\sum_{s^-\in S^-}{exp(sim(s_u^{a_i},s^-))}}
$$
**数据增强(crop)**: $s_u^{crop} = a_{crop}(s_u)=[v_c, v_{c+1},...,v_{c+L_c-1}]$ 其中$L_c=\lfloor\eta *|s_u|\rfloor$ 就是一个通过$\eta$计算的定长用户子序列. 这个方法可以提供一个用户的历史序列的local view, 并且如果不同crop结果有交集就是要相似最大化这部分交集, 如果没有交集则等价于序列推荐的序列预测任务.

**物品掩码(Item mask)**: $s_u^{mask}=a_{mask}(s_u)=[\hat{v_1}, \hat{v_2}, ...,\hat{v_{|s_u|}}]$ 其中<img src="https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220830115741510.png" alt="image-20220830115741510" style="zoom:35%;" />$\mathcal T_{s_u}=(t_1,t_2,...,t_{L_m}), L_m=\lfloor\gamma * |s_u|\rfloor$ ,  $t_j$代表在序列中将被掩盖的index下标, 物品掩码用于避免过拟合, 被掩码覆盖的结果和原序列在主要意图上应该是保留的.

**物品重拍(Item Reorder)**: $s_u^{reorder} = a_{reorder}(s_u)=[v_1, v_2, ..., \hat{v_i},...,\hat{v}_{i+L_r-1},...,v_{|s_u|}]$ 其中 $L_r=\lfloor\beta*|s_u|\rfloor$ ,从下标r开始的$L_r$个物品重排序

用户特征使用的[SASRec](./Self-Attentive_Sequential_Recommendation.md)架构, 单向Transformer, 如下图所示, 具体公式不做过多介绍, 和SASRec一样

![image-20220830122057780](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20220830122057780.png)

最后是**多任务训练**: 将对比学习的损失和序列推荐的损失线性相加就是
$$
\mathcal L_{total}=\mathcal L_{main} + \lambda\mathcal L_{cl} \\
\mathcal L_{main}(s_u,t)=-log\frac{exp(s_{u,t}^T v_{t+1}^+)}{exp(s_{u,t}^T v_{t+1}^+)+\sum_{v^-_{t+1}\in\mathcal V^-}exp(s_{u,t}^T v_{t+1}^-)}
$$
**实验结果**:  

重现回来补

**个人总结**：  

**备注**  