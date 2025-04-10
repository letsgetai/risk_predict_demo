import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse
import numpy as np
from torch.utils.data import DataLoader, Dataset
import nni
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 多头自注意力层
        self.attention = nn.MultiheadAttention(embed_dim=args.hidden_size, num_heads=args.attention_head, dropout=args.attention_dropout,batch_first=True)
        # 前馈层
        self.fc = nn.Linear(args.hidden_size, args.hidden_size)
        # Dropout层
        self.dropout = nn.Dropout(args.dropout)
        # 激活函数
        self.activation = nn.ReLU()

        self.norm = nn.LayerNorm(args.hidden_size)
    def forward(self, x):
        # x: [batch_size, patch_num, hidden_size]    
        # 多头自注意力
        attn_output, _ = self.attention(x, x, x)   
        # 残差连接和层归一化
        x = x + attn_output
        x = self.norm(x)
        # 通过前馈层
        z = self.fc(x)       
        x = self.norm(z+x)
        # Dropout和激活函数
        x = self.dropout(x)
        x = self.activation(x)        
        return x
class mlp(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer = nn.Linear(args.hidden_size, args.hidden_size)
        # Dropout层
        self.dropout = nn.Dropout(args.dropout)
        # 激活函数
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(args.hidden_size)
    def forward(self, x):
        # x: [batch_size, patch_num, hidden_size]
        z = self.layer(x)
        # Dropout和激活函数
        z = self.norm(z)
        z = self.dropout(z)
        x = self.activation(z)+x
        return x


# 定义MLP模型
class patch_mlp(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.hidden_layers = args.hidden_layers
        self.output_size = args.output_size
        self.head_size = args.head_size
        self.hidden_size= args.hidden_size

        self.patch_length = args.patch_length

        self.dropout = args.dropout
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay

        if args.input_dim % self.patch_length != 0:
            self.padding_len = self.patch_length - (args.input_dim % self.patch_length)
            self.num_patches = (args.input_dim + self.padding_len) // self.patch_length
        else:
            self.num_patches = args.input_dim // self.patch_length
            self.padding_len = 0
        self.embedding = nn.Linear(args.patch_length, self.hidden_size)
        if args.model:
            self.backbone = nn.Sequential(*[mlp(args) for _ in range(self.hidden_layers)])
        else:
            self.backbone = nn.Sequential(*[AttentionLayer(args) for _ in range(self.hidden_layers)])

        self.head_way = args.head_way
        if self.head_way == 1:
            self.head = nn.Linear(self.hidden_size*self.num_patches, self.output_size)
        elif self.head_way == 2:
            self.head1 = nn.Linear(self.hidden_size, 1)
            self.head2 = nn.Linear(self.num_patches, 1)
        elif self.head_way == 3:
            self.head1 = nn.Linear(self.hidden_size*self.num_patches, self.hidden_size*self.num_patches//2)
            self.head2 = nn.Linear(self.hidden_size*self.num_patches//2, self.hidden_size*self.num_patches//4)
            self.head3 = nn.Linear(self.hidden_size*self.num_patches//4, self.output_size)
            

        pos_weight=torch.tensor([args.pos_weight], dtype=torch.float32)  
        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss(pos_weight)  # 二分类任务，适用于信用风险预测
        def int_list():
            self.training_step_outputs = []
            self.train_loss=0.6
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
        x = self.add_noise(x)
        # x: [batch_size, input_dim]
        # 如果需要进行 padding
        if self.padding_len > 0:
            # 对输入进行 zero padding，使其长度为 patch_length 的倍数
            x = F.pad(x, (0, self.padding_len))  # 填充到 [batch_size, input_dim + padding]
        x = x.view(x.size(0), self.num_patches, self.patch_length)
        x = self.embedding(x)## x: [batch_size, patch_num, hidden_size]
        x = self.backbone(x)## x: [batch_size, patch_num, hidden_size]
        if self.head_way == 1:
            x = self.head(x.view(x.size(0), -1))## x: [batch_size, output_size]
        elif self.head_way == 2:
            x = self.head1(x).squeeze()    #x: [batch_size, patch_num]
            x = self.head2(x)    #x: [batch_size, 1]
        elif self.head_way == 3:
            x = self.head1(x.view(x.size(0), -1))    #x: [batch_size, hidden_size*patch_num]
            x = self.head2(x)    #x: [batch_size, hidden_size*patch_num//2]
            x = self.head3(x)    #x: [batch_size, output_size]
        return x
    
    def add_noise(self, x, noise_level=0.3):
        noise = noise_level * torch.randn_like(x)
        return x + noise

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,weight_decay=self.weight_decay )
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
        'val_recall': recall,
        'val_accuracy': accuracy,
        'val_loss': loss,
        'train_loss': self.train_loss       
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

        nni.report_final_result(accuracy+0.8*recall)

        self.test_correct=0
        self.test_total=0
        self.test_true_positives = 0
        self.test_false_negatives = 0

        tensorboard = self.logger.experiment
        tensorboard.add_text('test_recall',f'{recall}')
        tensorboard.add_text('test_accuracy',f'{accuracy}')


        








