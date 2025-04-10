import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
import argparse
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np  
import torch

class predict_dataset(Dataset):
    def __init__(self, args,scaler,features_to_remove,imputer):
        # 在这里加载和处理数据
        data = pd.read_csv(args.test_data_path)
        data = data.drop(columns=features_to_remove)
        data = imputer.transform(data)
        self.data = scaler.transform(data) 
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # 返回某一行的数据和标签
        row = self.data[idx]
        features = row[:]
        return features

# 示例自定义数据集
class CreditRiskDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # 返回某一行的数据和标签
        row = self.data[idx]
        features = row[:-1]
        label = row[-1]
        return features, label

# 定义LightningDataModule类
class CreditRiskDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args=args
        self.data_path = args.data_path
        self.traget_path = args.traget_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.scaler = None  # 用于标准化的数据缩放器
        self.prepare_train_data()

        
    def prepare_train_data(self):
        # 加载数据
        data = pd.read_csv(self.data_path)
        target=pd.read_csv(self.traget_path)
        # 处理缺失值过多特征
        missing_counts = data.isnull().sum()
        self.features_to_remove = missing_counts[missing_counts > self.args.threshold].index.tolist()    
        data = data.drop(columns=self.features_to_remove)
        #众数填充
        self.imputer = SimpleImputer(strategy='most_frequent')
        data = self.imputer.fit_transform(data)
        
        data = np.hstack((data, target['y'].values.reshape(-1, 1)))

        # 标准化
        self.scaler = StandardScaler()
        data[:, :-1] = self.scaler.fit_transform(data[:, :-1])  # 只对特征进行标准化

        # 保存处理后的数据
        self.processed_data = data
        self.random_split_dataset()

    def random_split_dataset(self):
        # 分割训练集、验证集和测试集
        dataset = CreditRiskDataset(self.processed_data)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        if self.args.weighted_sampler:
            labels = torch.tensor([self.processed_data[i][-1] for i in self.train_dataset.indices]).long()  # 获取训练集样本的标签
            
            class_counts = torch.bincount(labels)  # 计算每个类别的样本数
            class_weights = 1. / class_counts.float()  # 为每个类别计算权重
            # 为每个样本分配权重
            sample_weights = class_weights[labels]
            sampler = WeightedRandomSampler(weights=sample_weights, 
                                    num_samples=len(sample_weights), 
                                    replacement=True) 
            return DataLoader(self.train_dataset,batch_size=self.batch_size,sampler=sampler, num_workers=self.num_workers)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):

        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        dataset=predict_dataset(self.args,self.scaler,self.features_to_remove,self.imputer)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)



