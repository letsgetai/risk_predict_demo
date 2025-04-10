import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse
import numpy as np
from torch.utils.data import DataLoader, Dataset
import nni

class Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding_layer = nn.Linear(args.input_dim, args.embedding_dim)
        self.hidden_size_layer = nn.Linear(1, args.hidden_size)
    def forward(self, x):
        # x: [batch_size, input_size]
        # embedding: [seq_len, input_size, embedding_dim]
        x = self.embedding_layer(x)
        x = self.hidden_size_layer(x.unsqueeze(-1))
        return x
# 定义MLP模型
class MLP(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.hidden_layers = args.hidden_layers
        self.output_size = args.output_size
        self.head_size = args.head_size
        self.embedding_dim = args.embedding_dim
        self.hidden_size= args.hidden_size

        self.dropout = args.dropout
        self.learning_rate = args.learning_rate

        self.embedding = Embedding(args)
        # 定义MLP的网络结构
        layers = []

        # 隐藏层
        for _ in range(self.hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout))

        layers.append(nn.Linear(self.hidden_size, self.head_size))
        self.head=nn.Linear(self.embedding_dim, self.output_size)
        self.model = nn.Sequential(*layers)

        # self.linear= nn.Linear(self.input_size, self.output_size)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()  # 二分类任务，适用于信用风险预测
        def int_list():
            self.training_step_outputs = []
            self.train_loss=100
            self.val_step_outputs = []
            self.val_correct=0
            self.val_total=0
            self.val_true_positives = 0
            self.val_false_negatives = 0
            self.test_step_outputs = []
            self.test_correct=0
            self.test_total=0
            self.test_true_positives = 0
            self.test_false_negatives = 0
            self.epoch_count = 0
        int_list()
    def forward(self, x):
        x = self.embedding(x)#[bacthsize, input_size, embedding_dim]
        x = self.model(x)
        x = x.mean(dim=-1)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y.float())
        self.log("train_loss", loss,on_step=True, on_epoch=False)
        # self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y.float())
        self.log("val_loss", loss,on_step=True, on_epoch=False)
        # self.log('val_loss', loss, on_step=False, on_epoch=True,sync_dist=True)
        self.val_step_outputs.append(loss)
        # 计算准确率
        probabilities = torch.sigmoid(y_hat)
        predicted_labels = (probabilities >= 0.5).long()
                # 计算TP和FN
        self.val_true_positives += ((predicted_labels == 1) & (y == 1)).sum().item()
        self.val_false_negatives += ((predicted_labels == 0) & (y == 1)).sum().item()
        self.val_correct += (predicted_labels == y).sum().item()
        self.val_total += y.size(0)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y.float())
        self.test_step_outputs.append(loss)
        # 计算准确率
        probabilities = torch.sigmoid(y_hat)
        predicted_labels = (probabilities >= 0.5).long()
        # 计算TP和FN
        self.test_true_positives += ((predicted_labels == 1) & (y == 1)).sum().item()
        self.test_false_negatives += ((predicted_labels == 0) & (y == 1)).sum().item()
        self.test_correct += (predicted_labels == y).sum().item()
        self.test_total += y.size(0)

        return loss

    def predict_step(self, batch):
        x = batch
        y_hat = self(x).squeeze()
        # y_hat = torch.sigmoid(y_hat)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_end(self):
        # 计算平均loss
        
        loss = 0.
        for out in self.training_step_outputs:
            loss += out.cpu().detach().item()
        loss /= len(self.training_step_outputs)
        self.log('avg_train_loss', loss,on_step=False,on_epoch=True,sync_dist=True )
        # print(f"Average training loss: {loss}")
        self.training_step_outputs = []
        print(f"Epoch {self.epoch_count}, Average training loss: {loss}")
        self.epoch_count+=1
        self.train_loss= loss


    def on_validation_epoch_end(self):
        # 计算平均loss
        loss = 0.
        for out in self.val_step_outputs:
            loss += out.cpu().detach().item()
        loss /= len(self.val_step_outputs)
        self.log('avg_val_loss', loss,on_step=False,on_epoch=True,sync_dist=True,logger=True)
        # print(f"Average validation loss: {loss}")
        self.val_step_outputs = []
        # 计算准确率
        accuracy=self.val_correct/self.val_total
        self.log('val_accuracy', accuracy,on_step=False,on_epoch=True,sync_dist=True,logger=True)
        # print(f"Validation accuracy: {accuracy}")
        # 计算召回率
        recall = self.val_true_positives / (self.val_true_positives + self.val_false_negatives + 1e-6)  # 防止除以零
        self.log('val_recall', recall, on_step=False, on_epoch=True, sync_dist=True, logger=True)

        nni.report_intermediate_result({
        'train_loss': self.train_loss,
        'val_loss': loss,
        'val_accuracy': accuracy,
        'val_recall': recall
        })
        print(f"Validation loss: {loss}",f"Validation recall: {recall}",f"Validation accuracy: {accuracy}")

        # 重置统计数据
        self.val_correct = 0
        self.val_total = 0
        self.val_true_positives = 0
        self.val_false_negatives = 0

    def on_test_epoch_end(self):
        # 计算平均loss
        loss = 0.
        for out in self.test_step_outputs:
            loss += out.cpu().detach().item()
        loss /= len(self.test_step_outputs)
        self.log('avg_test_loss', loss, logger=True, sync_dist=True)
        self.test_step_outputs = []
        recall = self.test_true_positives / (self.test_true_positives + self.test_false_negatives + 1e-6)  # 防止除以零
        self.log('test_recall', recall, logger=True, sync_dist=True)
        # 计算准确率
        accuracy=self.test_correct/self.test_total
        self.log('test_accuracy', accuracy, logger=True, sync_dist=True)

        nni.report_final_result(loss)

        self.test_correct=0
        self.test_total=0
        self.test_true_positives = 0
        self.test_false_negatives = 0

        tensorboard = self.logger.experiment
        tensorboard.add_text('test_recall',f'{recall}')
        tensorboard.add_text('test_accuracy',f'{accuracy}')


        








