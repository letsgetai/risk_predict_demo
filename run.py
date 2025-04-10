import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import argparse
import os
import torch
import random
import numpy as np
from data_load import CreditRiskDataModule
from pacth_mlp import patch_mlp
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import pandas as pd
import os
import nni
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pytorch_lightning.callbacks import EarlyStopping


parser = argparse.ArgumentParser(description='link prediction')
# random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')
# basic config
# 数据路径、batch size和其他必要参数
parser.add_argument('--data_path', type=str, default='risk_control/data/训练集/train_data.csv', help='Path to the dataset')
parser.add_argument('--test_data_path', type=str, default='risk_control/data/测试集/test_data.csv', help='Path to save results')
parser.add_argument('--traget_path', type=str, default='risk_control/data/训练集/train_target.csv', help='Path to save results')
parser.add_argument('--result_path', type=str, default='', help='Path to save results')


parser.add_argument('--batch_size', type=int, default=512, help='Batch size for dataloaders')
parser.add_argument('--num_workers', type=int, default=120, help='Number of workers for DataLoader')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for optimizer')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--weighted_sampler', type=bool, default=True, help='sample')
parser.add_argument('--patience', type=int, default=3, help='')

parser.add_argument('--input_dim', type=int, default=3807, help='字典大小为特征总数')
parser.add_argument('--model', type=bool, default=False, help='ture使用mlp，flase是attention')
parser.add_argument('--head_way', type=int, default=3, help='1是一次直接降下来，2是先降patch_len再降num_patches,3是先和1一样合并，但是分段降')

# parser.add_argument('--norm_type', type=bool, default=True, help='True是BatchNorm1d，False是 LayerNorm1d')
parser.add_argument('--pos_weight', type=int, default=2, help='正样本损失关注度')
parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减大小，l2正则化')
parser.add_argument('--threshold', type=int, default=40000, help='删掉缺失值过多的特征')

parser.add_argument('--patch_length', type=int, default=128, help='patch_len')
parser.add_argument('--hidden_layers', type=int, default=3, help='Number of hidden layers')
parser.add_argument('--hidden_size', type=int, default=128, help='Number of units in each hidden layer')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
parser.add_argument('--output_size', type=int, default=1, help='Output size (for binary classification)')
parser.add_argument('--attention_head', type=int, default=4, help='attention head')
parser.add_argument('--attention_dropout', type=float, default=0.4, help='attention dropout')


parser.add_argument('--embedding_dim', type=int, default=128, help='mlp embedding里面用了，patch里面没用')
parser.add_argument('--head_size', type=int, default=128, help='pacth里面没有使用')

parser.add_argument('--devices', type=list, default=[0], help='GPU device to use')

args = parser.parse_args()
# 更新参数
argsparams = vars(args)
params = nni.get_next_parameter()
argsparams.update(params)
args = argparse.Namespace(**argsparams)
# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

data=CreditRiskDataModule(args)
args.input_dim = data.processed_data.shape[1]-1
print('输入特征个数：',args.input_dim)
filename = f"hidsize_{args.hidden_size}_dp_{args.dropout}_wight_decay_{args.weight_decay}_pos_weight_{args.pos_weight}_hlayers_{args.hidden_layers}_patchlen_{args.patch_length}_headway_{args.head_way}_bs_{args.batch_size}_lr_{args.learning_rate}_epoch_{args.num_epochs}_seed_{args.random_seed}_attnhead_{args.attention_head}_attndp_{args.attention_dropout}"
TensorBoardlogger = TensorBoardLogger( name=f"{filename}",save_dir=f"risk_control/logs/tensor")
CSVlogger = CSVLogger(name="my_model", save_dir="risk_control/logs/csv")
torch.set_float32_matmul_precision('high')
model = patch_mlp(args)

# 定义早停回调
early_stopping = EarlyStopping(
    monitor='avg_val_loss',  # 监控的指标，比如验证集的损失
    patience=args.patience,  # 如果连续3个epoch指标不再改善，停止训练
    verbose=True,  # 打印日志
    mode='min'  # 监控的是要减少的指标（例如损失）
)

trainer = Trainer(max_epochs=args.num_epochs,
                    accelerator='cuda',
                    devices=[0],
                    precision='64-true',
                    # strategy='ddp_find_unused_parameters_true',
                    logger=[TensorBoardlogger,CSVlogger],
                    callbacks=[early_stopping],
                    enable_progress_bar=False,
                    num_sanity_val_steps=0 )
trainer.fit(model,data)
trainer.test(model,data,ckpt_path='best')   



predict_dataloader= data.predict_dataloader()
predictions = trainer.predict(dataloaders=predict_dataloader,ckpt_path='best')
all_predictions = torch.cat(predictions).unsqueeze(1)
df = pd.DataFrame(all_predictions.numpy(), columns=['predictions'])
df_test=pd.read_csv(args.test_data_path)
result=pd.DataFrame()
result['idx']=df_test['idx']
result['y_pred']=df['predictions']
result.to_csv(f'risk_control/logs/tensor/{filename}/result.csv', index=False)


torch.cuda.empty_cache()
