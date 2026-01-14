# -*- coding: utf-8 -*-
"""
数据划分脚本，将原始JSON数据划分为训练集、验证集和测试集
"""
import os
import sys

# 将项目根目录添加到sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import random
from sklearn.model_selection import train_test_split
from config import config


def load_raw_data(raw_data_path):
    """
    加载原始JSON数据
    :param raw_data_path: 原始数据路径
    :return: 数据列表
    """
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    划分数据为训练集、验证集和测试集
    :param data: 原始数据
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    :param test_ratio: 测试集比例
    :param seed: 随机种子
    :return: 训练集、验证集、测试集
    """
    # 设置随机种子
    random.seed(seed)
    
    # 第一次划分：训练集 + 验证集+测试集
    train_data, temp_data = train_test_split(data, test_size=val_ratio + test_ratio, random_state=seed)
    
    # 第二次划分：验证集 + 测试集
    val_data, test_data = train_test_split(temp_data, test_size=test_ratio/(val_ratio + test_ratio), random_state=seed)
    
    return train_data, val_data, test_data


def save_split_data(train_data, val_data, test_data, processed_dir):
    """
    保存划分后的数据
    :param train_data: 训练集
    :param val_data: 验证集
    :param test_data: 测试集
    :param processed_dir: 处理后数据目录
    """
    # 保存路径
    train_path = os.path.join(processed_dir, 'train.json')
    val_path = os.path.join(processed_dir, 'val.json')
    test_path = os.path.join(processed_dir, 'test.json')
    
    # 保存数据
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据划分完成！")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    print(f"测试集: {len(test_data)} 条")
    print(f"保存路径: {processed_dir}")


def main():
    """
    主函数
    """
    # 加载原始数据
    data = load_raw_data(config.raw_data_path)
    print(f"原始数据总量: {len(data)} 条")
    
    # 划分数据
    train_data, val_data, test_data = split_data(
        data, 
        train_ratio=config.train_ratio, 
        val_ratio=config.val_ratio, 
        test_ratio=config.test_ratio, 
        seed=config.seed
    )
    
    # 保存划分后的数据
    save_split_data(train_data, val_data, test_data, config.processed_dir)


if __name__ == '__main__':
    main()