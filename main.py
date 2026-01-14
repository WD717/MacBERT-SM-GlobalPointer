# -*- coding: utf-8 -*-
"""
主入口文件，处理命令行参数，调用训练、评估或推理功能"""
import os
import sys
import json
import argparse

# 将当前目录添加到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config


def parse_args():
    """
    解析命令行参数
    :return: 命令行参数
    """
    parser = argparse.ArgumentParser(description='中文医疗实体识别模型')
    
    # 模型选择
    parser.add_argument('--model', type=str, default='bert_crf', help='模型名称')
    
    # 功能选择
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--eval', action='store_true', help='评估模型')
    parser.add_argument('--infer', action='store_true', help='推理文本')
    
    # 推理参数
    parser.add_argument('--text', type=str, default='', help='推理文本')
    
    # 配置文件
    parser.add_argument('--config', type=str, default='', help='配置文件路径')
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 更新配置
    config.update(model_name=args.model)
    
    print(f"\n中文医疗实体识别模型")
    print(f"模型名称: {config.model_name}")
    print(f"设备: {config.device}")
    
    # 1. 数据准备
    print(f"\n1. 数据准备")
    
    # 检查原始数据是否存在
    if not os.path.exists(config.raw_data_path):
        print(f"错误: 原始数据文件不存在: {config.raw_data_path}")
        return
    
    # 检查是否已经划分数据
    train_path = os.path.join(config.processed_dir, 'train.json')
    val_path = os.path.join(config.processed_dir, 'val.json')
    test_path = os.path.join(config.processed_dir, 'test.json')
    
    if not all([os.path.exists(train_path), os.path.exists(val_path), os.path.exists(test_path)]):
        print(f"正在划分数据...")
        # 运行数据划分脚本
        from data.split_data import main as split_main
        split_main()
    else:
        print(f"数据已划分完成")
    
    # 2. 执行功能
    if args.train:
        print(f"\n2. 开始训练模型")
        # 运行训练脚本
        from train import main as train_main
        train_main()
    
    if args.eval:
        print(f"\n2. 开始评估模型")
        # 运行评估脚本
        from evaluate import main as eval_main
        eval_main()
    
    if args.infer:
        print(f"\n2. 开始推理")
        # 运行推理脚本
        from infer import main as infer_main
        
        # 如果提供了文本参数，更新配置
        if args.text:
            # 修改infer.py中的默认文本
            import infer
            infer.text = args.text
        
        infer_main()
    
    if not any([args.train, args.eval, args.infer]):
        print(f"\n请指定功能: --train 训练模型, --eval 评估模型, --infer 推理文本")
        print(f"示例: python main.py --model bert_crf --train")
        print(f"示例: python main.py --model bert_crf --infer --text '【药品商品名称】 乌鸡白凤丸'")


if __name__ == '__main__':
    main()