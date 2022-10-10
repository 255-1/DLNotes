+ ***[Sequential Recommendation with Multiple Contrast Signals(ContraRec)](https://dl.acm.org/doi/abs/10.1145/3522673)***   

![image-20221010174515486](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20221010174515486.png)

**年份**：  2022

**引用次数**： 5

**应用领域**：  对比学习

**方法及优缺点**：

使用了两种对比context-target和context-context, 前一种希望序列和目标物品相似, 后一种希望相同物品的前置序列要尽量相似. 除此以外, context-target的方法可以看成是传统BPR的一种推广作者这里提出了BPR+新的loss. 	

**动机**:  

**相关工作和理论**：  

### CTC(Context-Target Contrast)

传统的推荐系统Loss的公式如下, 可以看出这个和对比学习的InfoNCE有非常相似的结构, 只是分母只有一个负样本
$$
\begin{equation*}
\begin{aligned}
\mathcal L_{BPR} &= \sum_{(S_t, i_t)}-\log\sigma(\hat y(S_t, i_t) - \hat y(S_t,i_t^-))\\
&= \sum_{(S_t, i_t)}-\log(\frac{1}{1+\exp(-\hat y(S_t, i_t) - \hat y(S_t,i_t^-))}) \\
&= \sum_{(S_t, i_t)}-\log(\frac{\exp (\hat y(S_t, i_t))}{\exp (\hat y(S_t, i_t))+\exp (\hat y(S_t, i_t^-))})
\end{aligned}
\end{equation*}
$$
推广得到CTC的Loss, 对不同的k个target作为负样本,
$$
\mathcal L_{CTC}=\tau_1\cdot\sum_{(S_t, i_t)}-log(\frac{exp(g(f(S_t),i_t/\tau_1)}{exp(g(f(S_t),i_t/\tau_1)+\sum_{k=1}^Kexp(g(f(S_t),i_k/\tau_1)}) \\
\mathcal L_{BPR+} = \sum_{(S_t, i_t)}-\frac{1}{K}\sum_{k}^K\log\sigma(\hat y(S_t, i_t)-\hat y(S_t, i_k))
$$

### CCC(Context-Context Contrast)

这里会使用数据增强的办法得到$2|\mathcal B|$个样本来 增强办法和$S^3$-Rec提出来的一样, $T(\cdot)$代表有一样item的序列
$$
L_{CCC} = \tau_2\cdot\sum_{\tilde S_t\in\mathcal A}\frac{1}{|T(\tilde S_t)|}\sum_{\tilde S_t'\in T(\tilde S_t)}l(f(\tilde S_t),f(\tilde S_t')) \\
l(f(\tilde S_t), f(\tilde S_t')) = -\log\frac{exp(sim(f(\tilde S_t), f(\tilde S_t'))/\tau_2)}{\sum_{\tilde S_t^-\in \mathcal A \backslash \tilde S_t} exp(sim(f(\tilde S_t), f(\tilde S_t^-))/\tau_2)}
$$


综上CTC和CCC得到伪代码如下

![image-20221010182935006](https://paperrecord.oss-cn-shanghai.aliyuncs.com/image-20221010182935006.png)

**实验结果**:  

**个人总结**：  

**备注**  



