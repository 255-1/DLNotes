+ ***[Representation Learning with Contrastive Predictive Coding(CPC)](https://arxiv.org/abs/1807.03748)***   

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081323657.PNG)    

**年份**：2018  

**引用次数**：2436  

**应用领域**：the basement of constrastive learning, unspervised learning  

**方法及优缺点**：  
    CPC作为特征提取可以用于任何领域,这些数据可以以有序的顺序表示: 文本、语音、视频,甚至图像(图像可以看作是一系列像素或补丁). CPC通过编码信息来学习表示,这些信息在多个时间步骤之间共享,而丢弃本地信息.这些特性通常被称为“慢特性”: 这些特性不会随着时间的推移而变化得太快.主要优点简单,计算需求小并且结果很encouraging, 而且这个用法可以用在许多形式.  

**结论**：
    本文提出了CPC架构用于提取紧凑的隐表征并用此编码预测未来的信息, CPC将自回归和NCE和预测编码的直觉相结合,使用无监督方法学习抽象的特征.  

**动机**:  
    无监督的预测未来的办法来自最古老的信息处理的数据压缩技术, 神经科学中,预测编码理论表明大脑可以预测不同的抽象层次上的结果. 也有论文实现了, 这些方法富有成效部分原因是因为我们预测相关值得上下文时会在一定条件下依赖于相同的共享的高级潜在信息.在时间序列模型例如RNN我们就是在不同时间步，利用信号的局部平滑性逐步预测，但随着预测步骤越多，共享的潜在信息就会越少，要建模更全局的结构，在这之中，慢性特征或成为我们的关注对象  
    本模型的主要思想是学习一个representation可以获得底层的共享的信息, 同时它摒弃了局部的low-level information和noise, 希望它能得到更加全局的特征(猜:类似Transformer). 其中有一大挑战是高维数据使用单峰的loss function如均方差和交叉熵并不好用, 而复杂的有条件能生成所有细节的生成模型又太过于复杂, 并且没有考虑上下文信息$c$, 进一步对于target $x$ 使用普通$p(x|c)$(分类常用的概率,在输入上下文的情况得到target的概率)没办法直接得到x与c之间户信息的最优解,由此引入了一个mutual information的公式
    $I(x,c)=\sum_{x,c}p(x,c)log\frac{p(x|c)}{p(x)}$(Eq.1)
    通过最大化这个互信息的公式就可以提取出的普遍的潜在变量.这里作者举的例子是图片标签是一个互信息,但是1000类图片的分类也就只要10bit就行,但是图片自己的bit信息有可能有上万,比如128x128x3, 希望找到更多的不只是10bit的互信息, 通过附录的证明可以得到后文的loss在减小的时候会提高互信息的下界, 所以不会被底层的冗余的互信息影响.   
**相关工作和理论**： 
    以下包含许多个人理解,不一定正确,本文主要完成了三件事情:  
    1)压缩了高维数据,降低了数据计算量  
    2)使用自回归去预测未来的几个step  
    3)使用NCE作为loss function训练了这个end-to-end模型    
设$z_t$是通过编码器$g_{enc}$得到的latent representations: $z_t=g_{enc}(x_t)$,这里的$x_t$又变成了音频的输入.接着是自回归模型$g_{ar}$总和了所有在当前时刻$t$之前的所有latent space,得到上下文的context latent representation $c_t=g_{ar}(z_{\leq t})$, Eq.1要做到最大化,对于$p(x,c)$无法控制,所以要最大化后面log里面的内容,作者单独提取出来设定为一个密度比density ratio: $\frac{p(x_{t+k}|c_t)}{p(x_{t+k})}$, 个人认为分子代表正样本,分母代表噪声,所以这个密度比越大越好. 作者认为一个相关性函数$f_k(x_{t+k}, c_t)$和密度比成比例(之后会解释), 说是任何positive real score都可以被用作$f_k$作者简单的设置了一个$f_k$函数使用了log-bilinear
$f_k(x_{t+k}, c_t) = exp(z^T_{t+k}W_kc_t)$  
作为相关性函数,这里e的幂应该是一个scalar,使用e估计是因为要用NCE. 通过使用$f_k$函数就不需要计算高维的概率分布,由此解决了1)的问题,并且可以借此通过抽样使用NCE等来评估
由此得到了NCE这里叫做InfoNCE,如下图公式所示:  
![InfoNCE](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081324482.PNG)  
减小InfoNCE就是增加log中的内容,就是增大分子减小分母, 增大就是增加相关性$f_k$,从而增加密度比,从而增大互信息.符合预期目标.
正样本来自条件概率分布的而不是来自先验概率分布的概率推导如下(缘由参考[InvaSpread](./Unsupervised_Embedding_Learning_via_Invariant_and_Spreading_Instance_Feature.md)):  
![InfoNCE](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081324145.PNG)  
由此得到的结果和前面的NCE的极大似然估计公式很像,所以这里认为$f_k$和密度比是成比例的.

encoder可以使用许多,作者用了renet block的 strided convolutional layers作为encoder, GRU作为自回归模型.说是自回归也可以用最新的self-attention或者masked convolutional architectures.  
**实验结果**:  
作者在音频,视觉,自然语言处理和强化学习领域做了一系列的对比实验. 等到需要复现的时候再来补充这一段内容.反正现在就是厉害就完了.
**个人总结**： 
以我现在就看过两篇对比学习的论文猜测,这篇文章提出了互信息用在对比学习可以用,并且用了提出了一系列的简化用法. 之前我看互信息是在一篇DeepInfoMax中,估计不是用在对比学习领域,CPC还是蛮难的文章,通过看别知乎的文章和google一些讨论才对文章稍微理解了一点(具体见论文目录中展示的看过的内容).  
**备注**:  
1)所谓mutual information就是引入了一个额外变量c使得x的变量的熵变低, 让x的确定性增加.  
    ![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081323769.svg)  

2)知乎有人说$f_k$是相似性,我感觉在公式理解上不太符合,我个人认为是相关性relevant,因为作者距离的公式中是$z_{t+k}$和$c_t$,如果说是相似性,相当于认为未来的特征要和当前的上下文一样.
