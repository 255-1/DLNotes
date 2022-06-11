+ ***[Graph Contextualized Self-Attention Network for Session-based Recommendation](https://www.ijcai.org/proceedings/2019/547)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202206111808289.PNG)

**年份**：  2019

**引用次数**： 170

**应用领域**： 序列推荐, GNN, 自注意力 

**方法及优缺点**：

使用Transformer代替了[SRGNN](./SRGNN.md)中的生成session部分, 其他一毛一样.按照作者的话就是Transformer可以学到更好的session global feature

**结论**：

同上.

**动机**:

作者就是想把Transformer放进来.  

**相关工作和理论**：  

序列图生成和GNN节点更新和[SRGNN](./SRGNN.md)中一样.

介绍Self-Attention Layer的内容, 其实和Transformer差不多. 假定GNN学习完了得到了所有物品的embedding$H = [h_1, h_2,...,h_n]$使用注意力机制得到session global feature $F$
$$
F = softmax(\frac{(HW^Q)(HW^K)^T}{\sqrt d})(HW^V)
$$
然后过一个两层的全连接层+参差层得到$E$
$$
E = ReLU(FW_1+b_1)W_2+b_2+F
$$
上述整合成$E = SAN(H)$,多个Self-Attention Layer堆叠到一起公式就是$E^{(k)}=SAN(E^{(k-1)})$.

在[SRGNN](./SRGNN.md)中global和local(最后一次点击) feature做的concate, 这里改为了权重求和, 合并公式为$S_f=\omega E_n^{(k)}+(1-\omega)h_n$, 最后预测就是和每个物品内积相似度过一个softmax,公式如下,最后loss function就是交叉熵带一个正则化结束
$$
\hat y_i=softmax(S_f^Tv_i).
$$
**实验结果**:  

复现回来补

**个人总结**：  

我都一眼看出来的究极水文, 居然有170引用, 甚至Trm都没有用多头

**备注**  