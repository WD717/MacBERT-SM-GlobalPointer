# -*- coding: utf-8 -*-
"""
模型评估脚本"""
import os
import sys

# 将项目根目录添加到sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import config
from utils.data_processor import DataProcessor, NERDataset
from train import ModelTrainer


def evaluate_model(model_name, test_data, processor, model_path):
    """
    评估模型
    :param model_name: 模型名称
    :param test_data: 测试数据
    :param processor: 数据处理器
    :param model_path: 模型路径
    """
    # 初始化训练器
    trainer = ModelTrainer(model_name, processor)
    
    # 加载模型
    trainer.load_model(model_path)
    
    # 创建测试数据集和数据加载器
    test_dataset = NERDataset(test_data, processor, mode=trainer.mode)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 评估模型
    metrics = trainer.evaluate(test_dataloader)
    
    # 输出评估结果
    print(f"\n模型: {model_name}")
    print(f"测试集评估结果:")
    print(metrics)
    
    return metrics


def main():
    """
    主函数
    """
    # 加载测试数据
    test_path = os.path.join(config.processed_dir, 'test.json')
    
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 加载标签映射
    label_mapping_path = os.path.join(config.processed_dir, 'label_mapping.json')
    processor = DataProcessor()
    processor.load_label_mapping(label_mapping_path)
    
    # 转换数据格式
    model_name = config.model_name
    if model_name in ['bert_crf', 'bert_bilstm_crf', 'macbert_bilstm', 'macbert_bilstm_crf', 'macbert_sdip_cnn_mlif_crf', 'macbert_crf']:
        # BIO格式
        test_data = processor.json_to_bio(test_data)
    else:
        # GlobalPointer格式
        test_data = processor.json_to_globalpointer(test_data)
    
    # 模型路径
    model_path = os.path.join(config.model_save_dir, f"{model_name}_best_model.pt")
    
    # 评估模型
    if os.path.exists(model_path):
        evaluate_model(model_name, test_data, processor, model_path)
    else:
        print(f"模型文件不存在: {model_path}")


if __name__ == '__main__':
    main()