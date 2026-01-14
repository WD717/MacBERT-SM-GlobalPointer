#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估修复后的GlobalPointer实现
"""

import sys
import os
import torch
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.base_model import GlobalPointer, GlobalPointerLoss
from models.bert_globalpointer import BertGlobalPointer
from utils.data_processor import DataProcessor
from utils.metrics import decode_globalpointer_logits, EntityMetrics
from config import config

def test_globalpointer_fix_eval():
    """评估修复后的GlobalPointer实现"""
    print("=== 评估修复后的GlobalPointer实现 ===")
    
    # 1. 准备数据
    print("1. 准备测试数据...")
    
    # 加载少量原始数据进行测试
    with open(config.raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)[:5]  # 只用前5个样本进行测试
    
    # 初始化数据处理器
    processor = DataProcessor()
    processor.collect_entity_types(raw_data)
    
    # 转换为GlobalPointer格式
    gp_data = processor.json_to_globalpointer(raw_data)
    
    # 创建小批量测试数据
    test_data = gp_data[:2]
    print(f"✓ 加载 {len(test_data)} 个测试样本")
    print(f"实体类型: {processor.entity_types}")
    
    # 2. 初始化模型
    print("\n2. 初始化模型...")
    num_entities = len(processor.entity_types)
    
    try:
        # 初始化模型（使用随机初始化的BERT）
        model = BertGlobalPointer(num_entities, model_path=None)
        print("✓ BERT-GlobalPointer模型初始化成功")
        
        # 3. 测试前向传播和损失计算
        print("\n3. 测试前向传播和损失计算...")
        
        # 准备一个批次的输入
        test_item = test_data[0]
        text = test_item['text']
        entities = test_item['entities']
        
        # 分词
        tokenized = processor.tokenizer(
            text,
            max_length=config.max_seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        token_type_ids = tokenized['token_type_ids']
        
        # 生成标签
        labels = processor.generate_globalpointer_labels(text, entities)
        labels = labels.unsqueeze(0)  # 添加batch维度
        
        # 前向传播
        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)
        print(f"✓ 前向传播成功，logits形状: {logits.shape}")
        
        # 计算损失
        loss_fn = GlobalPointerLoss()
        loss = loss_fn(logits, labels)
        print(f"✓ 损失计算成功，损失值: {loss.item():.4f}")
        
        # 4. 测试解码
        print("\n4. 测试实体解码...")
        
        # 解码实体
        decoded_entities = decode_globalpointer_logits(logits, attention_mask, threshold=0.7)
        print(f"✓ 解码成功，预测实体数量: {len(decoded_entities[0])}")
        print(f"真实实体数量: {len(entities)}")
        
        # 打印实体
        print("\n真实实体:")
        for start, end, label_id in entities:
            entity_text = text[start:end+1]
            print(f"  {entity_text} ({start}-{end}): {label_id}")
        
        print("\n预测实体:")
        for start, end, label in decoded_entities[0]:
            print(f"  [{start}-{end}]: {label}")
        
        # 5. 测试不同阈值
        print("\n5. 测试不同阈值下的实体预测...")
        thresholds = [0.3, 0.5, 0.7, 0.9]
        for th in thresholds:
            entities = decode_globalpointer_logits(logits, attention_mask, threshold=th)
            num_entities = len(entities[0])
            print(f"  阈值 {th}: {num_entities} 个实体")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_globalpointer_fix_eval()
    if success:
        print("\n✅ 修复后的GlobalPointer评估通过!")
        sys.exit(0)
    else:
        print("\n❌ 修复后的GlobalPointer评估失败!")
        sys.exit(1)
