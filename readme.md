
# 特征
数据包括客户基本信息类(x0-x20)，征信数据-历史金融借贷类(x20-x256)、征信数据-其他行为类(x256-x3805)风险标签:客户在授信后是否发生逾期，其中y=1代表逾期，y=0代表未逾期。
时间变量:变量date记录了客户的授信所属的阶段。

# 样本量
总计8万，其中训练集数据量6.2万，测试集数据量1.8万。训练集正样本(y=1)约占比18%。测试集正样本占比11%。

![alt text](png/image.png)

![alt text](png/image-1.png)

![alt text](png/image111.png)

根据路径打开tensorboard的命令
tensorboard --logdir=risk_control/logs/tensor/


目标：预测y=1的概率
对于模型输出结果，需要处理为0-1之间的概率值
是在每个batch后面加，还是在全部结果后面加？

## 欠采样
~~之前做的有一点问题，采样完之后整个结果都会变得特别小，由于经过sigmoid，此时可能模型预测的logit全是特别大的负数，但是取消的来这个采样，结果只是好了一点点，但是仍然是负数~~
** 需要确认一下这么分训练里面有多少正样本**
## 模型结构
> 只加一个线性层似乎结果正常，因为模型太过复杂，可能会对负样本过拟合

考虑分块让模型学到更有信息的特征

## pyl
data类里面的set_up,prepare_data在trainer.fit里面会自动调用，不要显示调用了
trainer.fit只会自动执行训练和验证，test需要自己在手动掉一次
log在tensorboard里面只有step的横坐标，但是会根据onstep还有onepoch画点

## nni参数搜索
yaml执行
nnictl create --config config.yaml --port 8080
nnictl stop p43ny6ew

| Category     | Brief Introduction                                                                                                                                                        |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TPE          | Tree-structured Parzen Estimator，经典的贝叶斯优化算法（参考论文）。TPE 是一个轻量级调优器，无需额外依赖，支持所有搜索空间类型，适合入门使用。缺点是TPE无法发现不同超参数之间的关系。 |
| Random       | 基础的随机搜索算法，作为基线。支持所有搜索空间类型。                                                                                                      |
| Grid Search  | 将搜索空间划分为均匀的网格，并执行暴力遍历搜索。另一种基线方法。支持所有搜索空间类型。适合在搜索空间较小时使用，或者需要严格找到最优超参数时使用。                                                       |
| Anneal       | 退火算法，此简单的退火算法开始时从先验分布中采样，随着时间推移趋向于从最优的点附近采样。此算法是随机搜索的一个变种，利用响应面的平滑性。退火率不可自适应。注意：Anneal需要通过命令安装`pip install nni[Anneal]`。   |
| Evolution    | 源自《大型图像分类器的进化》(参考论文)。该算法根据搜索空间随机初始化一个种群，每一代选择表现较好的个体，并对它们进行变异（例如改变超参数，添加/删除一层）来生成下一代。Evolution需要大量的试验，但其简单且易于扩展新特性。 |
| SMAC         | 基于Sequential Model-Based Optimization (SMBO) 的贝叶斯优化算法。SMAC 通过引入随机森林模型来处理分类参数。NNI 中的SMAC 是对SMAC3 GitHub仓库的封装（参考论文，GitHub仓库）。需要安装：`pip install nni[SMAC]`。 |
| Batch        | Batch tuner 允许用户为其试验代码提供多个配置（即超参数的选择），在完成所有配置后，实验结束。Batch tuner 只支持搜索空间中的choice类型。                                                                      |
| Hyperband    | Hyperband 尝试使用有限的资源探索尽可能多的配置，并返回最有前途的配置作为最终结果。其基本思想是生成许多配置并运行少量试验，淘汰一半最不具前景的配置，对剩余的进行进一步训练，并选择一些新的配置（参考论文）。             |
| Metis        | Metis 在调优参数时提供两个输出：（a）当前预测的最佳配置，（b）下一次试验的建议。它还能告知某些超参数是否需要重新采样（参考论文）。                                                                  |
| BOHB         | BOHB 是Hyperband的后续工作，针对其随机生成新配置的弱点，利用已完成的试验构建多个TPE模型，通过这些模型生成一部分新的配置（参考论文）。                                             |
| GP           | Gaussian Process Tuner，基于高斯过程的顺序模型优化方法（参考论文，GitHub仓库）。                                                                        |
| PBT          | Population Based Training (PBT) 是一种简单的异步优化算法，有效利用固定的计算预算来联合优化一群模型及其超参数，以最大化性能（参考论文）。                                                       |
| DNGO         | DNGO 使用神经网络代替高斯过程，用于建模函数分布的贝叶斯优化方法。                                                                                                           |

`report_intermediate_result`一个epoch只能报告一次，如果第二次报告不同的键对就覆盖上一次的，而且会产生隔一个横坐标画图

`experiment.config.experiment_working_directory='/home/liubin/huquan/LLM/risk_control/nni_experiment'`log文件的地址设置



pacth,开始（batch，feature）--->（batch，num_patch,patch_len),然后应该是要对pacth_len进行embedding，变成（bacth,num_pacth,hidden_size）,然后应该是要提取特征，多多尝试一下吧，这里可以用mlp提取，也可以用transformer提取，进好几个提取特征的层，然后就需要head，然后变成（batch，num_pacth*hidden_size）,过一个线性层，或者好几个线性层变成（bacth，1）

准确 召回
baseline 0.6334841847419739  0.5514512062072754 
调高去掉特征的阈值：0.6179702877998352     0.5734388828277588 （50000）0.5895281434059143    0.6358839273452759（较好）
pacth-len 128会较好
hidden——size 似乎无明显变化，都是59，256效果反而还不好了