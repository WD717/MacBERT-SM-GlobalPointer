# -*- coding: utf-8 -*-
"""
配置文件，定义所有超参数和路径
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 
import torch

class Config:
    def __init__(self):
        # 基本配置
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.project_dir, 'data')
        self.raw_data_path = os.path.join(self.data_dir, 'raw', 'medical_ner_entities.json')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        self.output_dir = os.path.join(self.project_dir, 'output')
        self.model_save_dir = os.path.join(self.output_dir, 'models')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        
        # 确保目录存在
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 模型配置
        self.model_name = 'bert_crf'  # 默认模型
        self.bert_model = 'bert-base-chinese'  # BERT模型路径
        self.macbert_model = 'hfl/chinese-macbert-base'  # MacBERT模型路径
        self.max_seq_len = 256  # 最大序列长度
        self.hidden_dim = 768  # BERT/MacBERT隐藏层维度
        
        # BiLSTM配置
        self.lstm_hidden_dim = 256  # LSTM隐藏层维度
        self.lstm_layers = 2  # LSTM层数
        self.lstm_dropout = 0.1  # LSTM dropout
        
        # CRF配置
        self.crf_lr = 2e-5  # CRF学习率
        
        # GlobalPointer配置
        self.gp_dim = 64  # GlobalPointer维度
        self.gp_threshold = 0.5  # GlobalPointer预测阈值
        
        # CNN配置
        self.cnn_filters = 64  # CNN卷积核数量
        self.cnn_kernel_sizes = [3, 4, 5]  # CNN卷积核大小
        
        # 训练配置
        self.train_batch_size = 4  # 训练批次大小
        self.eval_batch_size = 8  # 评估批次大小
        self.learning_rate = 2e-5  # 学习率
        self.weight_decay = 0.01  # 权重衰减
        self.epochs = 10  # 训练轮数
        self.early_stopping_patience = 3  # 早停轮数
        self.warmup_steps = 10  # 热身步数
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_gpu = torch.cuda.device_count()  # GPU数量
        
        # 数据配置
        self.train_ratio = 0.8  # 训练集比例
        self.val_ratio = 0.1  # 验证集比例
        self.test_ratio = 0.1  # 测试集比例
        self.seed = 42  # 随机种子
        
        # 推理配置
        self.infer_threshold = 0.5  # 推理阈值
        
        # 其他配置
        self.logging_steps = 100  # 日志记录步数
        self.save_steps = 100  # 模型保存步数
        
    def update(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

# 创建全局配置实例
config = Config()