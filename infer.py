# -*- coding: utf-8 -*-
"""
模型推理脚本"""
import os
import sys

# 将项目根目录添加到sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import torch
from config import config
from utils.data_processor import DataProcessor
from train import ModelTrainer


def infer_text(model_name, text, processor, model_path):
    """
    推理文本
    :param model_name: 模型名称
    :param text: 输入文本
    :param processor: 数据处理器
    :param model_path: 模型路径
    :return: 实体列表
    """
    # 初始化训练器
    trainer = ModelTrainer(model_name, processor)
    
    # 加载模型
    trainer.load_model(model_path)
    
    # 推理
    entities = trainer.predict(text)
    
    return entities


def main():
    """
    主函数
    """
    # 加载标签映射
    label_mapping_path = os.path.join(config.processed_dir, 'label_mapping.json')
    processor = DataProcessor()
    processor.load_label_mapping(label_mapping_path)
    
    # 模型路径
    model_name = config.model_name
    model_path = os.path.join(config.model_save_dir, f"{model_name}_best_model.pt")
    
    # 输入文本
    text = "【药品商品名称】 乌鸡白凤丸 【药品名称】 乌鸡白凤丸 【适应症】 补气养血，调经止带。用于气血两虚，身体瘦弱，腰膝酸软，月经不调，带下。"
    
    # 推理
    if os.path.exists(model_path):
        entities = infer_text(model_name, text, processor, model_path)
        print(f"\n输入文本: {text}")
        print(f"识别结果:")
        for entity in entities:
            print(f"- {entity['label']}: {entity['entity']} (位置: {entity['start_offset']}-{entity['end_offset']})")
    else:
        print(f"模型文件不存在: {model_path}")


if __name__ == '__main__':
    main()