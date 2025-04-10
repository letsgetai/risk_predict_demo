# 金融风控模型 - 客户逾期预测

![Project Banner](png/image.png)

## 📊 数据集说明

### 数据特征
| 特征类别 | 变量范围 | 描述 |
|---------|---------|------|
| 客户基本信息 | x0-x20 | 包含客户 demographic 信息 |
| 历史金融借贷 | x20-x256 | 征信系统中的借贷历史记录 |
| 其他行为数据 | x256-x3805 | 客户其他金融/消费行为 |

**目标变量**：  
`y=1` 表示授信后发生逾期，`y=0` 表示正常还款

### 数据统计
| 数据集 | 样本量 | 正样本占比 | 时间维度 |
|-------|-------|----------|--------|
| 训练集 | 62,000 | 18% | 分阶段记录 (`date`变量) |
| 测试集 | 18,000 | 11% | 与训练集同分布 |

![数据分布可视化](png/image-1.png)
*图：训练集与测试集的样本分布*

## 🎯 建模目标
预测客户逾期概率 `P(y=1|x) ∈ [0,1]`

**输出处理建议**：
```python
# 模型输出后处理方案
prob = torch.sigmoid(logits)  # 确保输出在0-1之间
🧠 模型架构
当前最佳结构
mermaid
复制
graph TD
    A[原始特征] --> B[Patch分割]
    B --> C[Embedding层]
    C --> D[特征提取块]
    D --> E[预测头]
    
    subgraph 特征提取
    D --> D1[MLP/Transformer]
    D1 --> D2[BatchNorm]
    end
超参数建议：

Patch长度: 128
Hidden size: 64-256 (需验证)
使用线性层+正则化防止过拟合
⚙️ 训练配置
关键参数
bash
复制
tensorboard --logdir=risk_control/logs/tensor/
注意事项
DataLoader 会自动调用 prepare_data()
测试集评估需手动执行：
python
复制
trainer.test(model)
日志记录区分 on_step 和 on_epoch
🔍 实验记录
采样策略对比
方法	准确率	召回率	备注
原始数据	0.633	0.551	Baseline
欠采样	0.618	0.573	需验证正样本量
阈值调整	0.590	0.636	最佳平衡
NNI超参搜索
bash
复制
nnictl create --config config.yaml --port 8080
算法对比表：

算法	优点	缺点	安装
TPE	轻量级贝叶斯优化	忽略参数关联	内置
SMAC	处理分类变量好	依赖随机森林	pip install nni[SMAC]
BOHB	组合Hyperband+TPE	计算成本高	内置
png/image111.png

🚀 使用指南
安装依赖：
bash
复制
pip install -r requirements.txt
启动训练：
bash
复制
python train.py --config configs/base.yaml
监控实验：
bash
复制
tensorboard --logdir=logs/
📌 待解决问题
 验证欠采样后的正样本比例
 优化patch-based特征提取层
 测试不同hidden_size的影响