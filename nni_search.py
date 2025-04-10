# 使用 nni.get_next_parameter() 获取需要评估的超参；

# 使用 nni.report_intermediate_result() 报告每个 epoch 产生的中间训练结果；

# 使用 nni.report_final_result() 报告最终准确率。
search_space = {
    'batch_size': {'_type': 'choice', '_value': [128, 256, 512]},
    # 'embedding_dim': {'_type': 'choice', '_value': [128, 256, 64,32,16,360,180,90]},
    'hidden_layers': {'_type': 'choice', '_value': [1,2,3]},
    'hidden_size': {'_type': 'choice', '_value': [64,32,16,128]},
    'learning_rate': {'_type': 'loguniform', '_value': [0.000001, 0.1]},
    'dropout': {'_type': 'uniform', '_value': [0.00001, 0.9999]},
    'patience': {'_type': 'choice', '_value': [ 3, 4, 5, 6]},
    'num_epochs': {'_type': 'choice', '_value': [10,15]},
    'attention_head': {'_type': 'choice', '_value': [1,2,4,8,16]},
    'attention_dropout': {'_type': 'uniform', '_value': [0.00001, 0.9999]},
    'model': {'_type': 'choice', '_value': [False]},
    'head_way': {'_type': 'choice', '_value': [1,2,3]},
    'patch_length': {'_type': 'choice', '_value': [16,32,36,64,96,128]},
    # 'weighted_sampler': {'_type': 'choice', '_value': [True]},
    'pos_weight': {'_type': 'choice', '_value': [2,3]},
    'weight_decay': {'_type': 'choice', '_value': [ 0.0001, 0.001, 0.01, 0.1, 1]},
    'threshold': {'_type': 'choice', '_value': list(range(0, 60001))}  # 新添加的阈值
}
import nni
from nni.experiment import Experiment

experiment = Experiment('local')
experiment.config.experiment_working_directory='/home/liubin/huquan/LLM/risk_control/nni_experiment'

experiment.config.trial_command = 'python risk_control/run.py'
experiment.config.trial_code_directory = '/home/liubin/huquan/LLM'

experiment.config.search_space = search_space

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.trial_concurrency = 1
experiment.max_experiment_duration = '20h' 
experiment.max_trial_number=300
experiment.run(8081)