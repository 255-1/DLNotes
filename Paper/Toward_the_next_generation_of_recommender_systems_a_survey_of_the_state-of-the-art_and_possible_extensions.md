+ ***[Toward the next generation of recommender systems: a survey of the state-of-the-art and possible extensions](https://ieeexplore.ieee.org/document/1423975)***   
**年份**：2005  
**引用次数**：9271  
**应用领域**：推荐系统  
**方法及优缺点**：
*1.Content-Based*: 特征限制(非文本信息难以提取特征),相同特征的物品无法分辨,OverSpecialization(即使有其他种类很好的物品也不会推荐 || 推荐太过相似的物品), 新用户加入没有打分  
*2.Collabrate Methods*: 新用户问题(可以用hybrid 推荐或者让用户给最利于推荐系统的物品评分)，新物品问题(没人打分也无法推荐，可以用hybrid推荐)，打分少(计算用户相似度不仅通过用户打分相似度也通过人口统计信息的相似度,还可以使用associative retrieval framework, related spreading activation计算用户相似度,可以使用SVD处理稀疏矩阵)  
**展望**：
> 1.改进users和items(利用历史信息)   
> 2.推荐过程中加入上下文信息(time, place etc. 提高维度)  
> 3.加入多标准评分(Pareto optimal solutions)  
> 4.更灵活(推荐内容是固定的,可以使用RQL或者使用OLAP-based)但是更少侵略性(可以通过用户浏览时间得到评分,但是不够准确,主动学习)的推荐过程  
> 5.需要更高效，并且拓宽应用邻域  
> 6.其他(explainability, trustworthiness, scalability and privacy)  


**相关工作**：
*1.Content-Based Methods*：信息检索的TF-IDF+cosine相似度, 朴素贝叶斯 $ P(C_i|k_{i,j}..k_{n,j})$ , 此外文本索引方面可以额外使用adaptive filtering(连续读入文档流使识别相关文档更加准确)和threshold setting(确定给定查询和文档程度)  
*2.Collaborative Methods*: 分为memery-based(heuristic-based)和model-based. memery-based中额外用perference-based filtering 计算用户打分的相对相关性,而不是绝对的分数数值(打分等级可能不同), 用户相似度使用Pearson系数(correlation-based)或者cosine相似度,此外还可以用default voting, inverse user frequency, case amplification, weighted-majority prediction计算用户相似度.计算用户相似度的方法可以用来计算物品相似度来获取物品评分,并且这种方法的推荐效果不错. model-based中使用贝叶斯模型缺陷只能把用户聚类到一个类别中，model-based的准确性经验上优于memory-based,无理论证明。还有使用了K-means，Gibbs sampling，probabilistic relational mode,a linear regression, 和 maximum entropy model以及更复杂的模型，Markov decision processes, latent semantic analysis。model和memory一起用效果更好。Collaborate Methods中可以使用去除噪声，冗余，充分利用用户评分的稀疏矩阵来提高准确度和处理大数据时的效率
*3.HyBrid Methods*: 物品加入协同,协同加入物品(用latent semantic indexing将物品向量转换成用户向量),以及将二者合二为一的模型(基于probabilisitc latent semantic analysis || Markov chain Monte Carlo methods). knowledgs-based techniques(doamin knowledge)经验上,混合效果比起单个更好.

![](https://paperrecord.oss-cn-shanghai.aliyuncs.com/202204081304807.png)
**个人总结**：本综述讲解推荐系统算法主要是讲文本相关领域，算法偏向机器学习,作为较早的综述。讲解了基本的content-based和collabrate-based的常用算法和区别以及各自的优缺点，提出了混合使用二者以及未来做出的扩展
