# -*- coding: utf-8 -*-
"""
模型训练脚本"""
import os
import sys

# 将项目根目录添加到sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from config import config
from utils.data_processor import DataProcessor, NERDataset
from utils.metrics import EntityMetrics, decode_bio_labels, decode_globalpointer_logits
from models.bert_crf import BertCRF
from models.bert_bilstm_crf import BertBiLSTMCRF
from models.bert_globalpointer import BertGlobalPointer
from models.macbert_bilstm_crf import MacBERTBiLSTMCRF
from models.macbert_globalpointer import MacBERTGlobalPointer
from models.macbert_mlif_globalpointer import MacBERTMLIFGlobalPointer
from models.macbert_sdip_cnn_globalpointer import MacBERTSDIPCNNGlobalPointer
from models.bert_sdip_cnn_mlif_globalpointer import BERTsdipCNNMLIFGlobalPointer
from models.macbert_sdip_cnn_mlif_crf import MacBERTSDIPCNNMLIFCRF
from models.macbert_sdip_cnn_mlif_globalpointer import MacBERTSDIPCNNMLIFGlobalPointer
from models.macbert_crf import MacBertCRF


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model_name, data_processor):
        """
        初始化模型训练器
        :param model_name: 模型名称
        :param data_processor: 数据处理器
        """
        self.model_name = model_name
        self.data_processor = data_processor
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = config.device
        
        # 模型映射
        self.model_map = {
            'bert_crf': self._init_bert_crf,
            'bert_bilstm_crf': self._init_bert_bilstm_crf,
            'bert_globalpointer': self._init_bert_globalpointer,
            'macbert_bilstm': self._init_macbert_bilstm,
            'macbert_bilstm_crf': self._init_macbert_bilstm_crf,
            'macbert_globalpointer': self._init_macbert_globalpointer,
            'macbert_mlif_globalpointer': self._init_macbert_mlif_globalpointer,
            'macbert_sdip_cnn_globalpointer': self._init_macbert_sdip_cnn_globalpointer,
            'bert_sdip_cnn_mlif_globalpointer': self._init_bert_sdip_cnn_mlif_globalpointer,
            'macbert_sdip_cnn_mlif_crf': self._init_macbert_sdip_cnn_mlif_crf,
            'macbert_sdip_cnn_mlif_globalpointer': self._init_macbert_sdip_cnn_mlif_globalpointer,
            'macbert_crf': self._init_macbert_crf
        }
        
        # 初始化模型
        self._init_model()
    
    def _init_bert_crf(self):
        """初始化BERT-CRF模型"""
        num_tags = len(self.data_processor.label2id)
        model = BertCRF(num_tags=num_tags, model_path=config.bert_model)
        return model, 'bio'
    
    def _init_bert_bilstm_crf(self):
        """初始化BERT-BiLSTM-CRF模型"""
        num_tags = len(self.data_processor.label2id)
        model = BertBiLSTMCRF(num_tags=num_tags, model_path=config.bert_model)
        return model, 'bio'
    
    def _init_bert_globalpointer(self):
        """初始化BERT-GlobalPointer模型"""
        num_entities = len(self.data_processor.entity_types)
        model = BertGlobalPointer(num_entities=num_entities, model_path=config.bert_model)
        return model, 'globalpointer'
    
    def _init_macbert_bilstm(self):
        """初始化MacBERT-BiLSTM模型"""
        num_tags = len(self.data_processor.label2id)
        model = MacBERTBiLSTM(num_tags=num_tags, model_path=config.macbert_model)
        return model, 'bio'
    
    def _init_macbert_bilstm_crf(self):
        """初始化MacBERT-BiLSTM-CRF模型"""
        num_tags = len(self.data_processor.label2id)
        model = MacBERTBiLSTMCRF(num_tags=num_tags, model_path=config.macbert_model)
        return model, 'bio'
    
    def _init_macbert_globalpointer(self):
        """初始化MacBERT-GlobalPointer模型"""
        num_entities = len(self.data_processor.entity_types)
        model = MacBERTGlobalPointer(num_entities=num_entities, model_path=config.macbert_model)
        return model, 'globalpointer'
    
    def _init_macbert_mlif_globalpointer(self):
        """初始化MacBERT-MLIF-GlobalPointer模型"""
        num_entities = len(self.data_processor.entity_types)
        model = MacBERTMLIFGlobalPointer(num_entities=num_entities, model_path=config.macbert_model)
        return model, 'globalpointer'
    
    def _init_macbert_sdip_cnn_globalpointer(self):
        """初始化MacBERT-SDIP-CNN-GlobalPointer模型"""
        num_entities = len(self.data_processor.entity_types)
        model = MacBERTSDIPCNNGlobalPointer(num_entities=num_entities, model_path=config.macbert_model)
        return model, 'globalpointer'
    
    def _init_bert_sdip_cnn_mlif_globalpointer(self):
        """初始化BERT-SDIP-CNN-MLIF-GlobalPointer模型"""
        num_entities = len(self.data_processor.entity_types)
        model = BERTsdipCNNMLIFGlobalPointer(num_entities=num_entities, model_path=config.bert_model)
        return model, 'globalpointer'
    
    def _init_macbert_sdip_cnn_mlif_crf(self):
        """初始化MacBERT-SDIP-CNN-MLIF-CRF模型"""
        num_tags = len(self.data_processor.label2id)
        model = MacBERTSDIPCNNMLIFCRF(num_tags=num_tags, model_path=config.macbert_model)
        return model, 'bio'
    
    def _init_macbert_sdip_cnn_mlif_globalpointer(self):
        """初始化MacBERT-SDIP-CNN-MLIF-GlobalPointer模型"""
        num_entities = len(self.data_processor.entity_types)
        model = MacBERTSDIPCNNMLIFGlobalPointer(num_entities=num_entities, model_path=config.macbert_model)
        return model, 'globalpointer'
    
    def _init_macbert_crf(self):
        """初始化MacBERT-CRF模型"""
        num_tags = len(self.data_processor.label2id)
        model = MacBertCRF(num_tags=num_tags, model_path=config.macbert_model)
        return model, 'bio'
    
    def _init_model(self):
        """
        初始化模型
        """
        if self.model_name not in self.model_map:
            raise ValueError(f"不支持的模型名称: {self.model_name}")
        
        # 初始化模型和模式
        self.model, self.mode = self.model_map[self.model_name]()
        self.model.to(self.device)
    
    def _init_optimizer(self, train_dataloader):
        """
        初始化优化器和学习率调度器
        :param train_dataloader: 训练数据加载器
        """
        # 模型参数
        param_optimizer = list(self.model.named_parameters())
        
        # 分组参数
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': config.weight_decay
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        # 优化器
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            eps=1e-8
        )
        
        # 学习率调度器
        total_steps = len(train_dataloader) * config.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
    
    def train_epoch(self, train_dataloader):
        """
        训练一个epoch
        :param train_dataloader: 训练数据加载器
        :return: 训练损失
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_dataloader, desc="训练"):
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播
            loss = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        return avg_loss
    
    def evaluate(self, eval_dataloader):
        """
        评估模型
        :param eval_dataloader: 评估数据加载器
        :return: 评估指标
        """
        self.model.eval()
        metrics = EntityMetrics(list(self.data_processor.entity_types))
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="评估"):
                # 将数据移到设备上
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                if self.mode == 'bio':
                    # 解码BIO标签
                    pred_entities = decode_bio_labels(outputs, self.data_processor.id2label, attention_mask)
                    true_entities = decode_bio_labels(labels, self.data_processor.id2label, attention_mask)
                else:
                    # 解码GlobalPointer结果
                    pred_entities = decode_globalpointer_logits(
                        outputs, attention_mask, 
                        threshold=config.gp_threshold,
                        entity_types=list(self.data_processor.entity_types)
                    )
                    # 真实标签需要转换为实体格式
                    batch_size = labels.shape[0]
                    true_entities_batch = []
                    for i in range(batch_size):
                        entities = []
                        for j, entity_type in enumerate(self.data_processor.entity_types):
                            # 找到真实实体
                            entity_mask = labels[i, j] == 1
                            for start in range(entity_mask.shape[0]):
                                for end in range(start, entity_mask.shape[1]):
                                    if entity_mask[start, end]:
                                        entities.append((start, end, entity_type))
                        true_entities_batch.append(entities)
                    true_entities = true_entities_batch
                
                # 更新指标
                for true_ent, pred_ent in zip(true_entities, pred_entities):
                    metrics.update(true_ent, pred_ent)
        
        return metrics
    
    def train(self, train_data, val_data):
        """
        训练模型
        :param train_data: 训练数据
        :param val_data: 验证数据
        """
        # 创建数据集和数据加载器
        train_dataset = NERDataset(train_data, self.data_processor, mode=self.mode)
        val_dataset = NERDataset(val_data, self.data_processor, mode=self.mode)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # 初始化优化器
        self._init_optimizer(train_dataloader)
        
        # 早停机制
        best_f1 = 0.0
        patience = 0
        
        for epoch in range(config.epochs):
            print(f"\nEpoch {epoch+1}/{config.epochs}")
            print("-" * 50)
            
            # 训练
            train_loss = self.train_epoch(train_dataloader)
            print(f"训练损失: {train_loss:.4f}")
            
            # 评估
            metrics = self.evaluate(val_dataloader)
            print(metrics)
            
            # 保存最佳模型
            current_f1 = metrics.compute()['micro']['f1']
            if current_f1 > best_f1:
                best_f1 = current_f1
                patience = 0
                # 保存模型
                model_save_path = os.path.join(config.model_save_dir, f"{self.model_name}_best_model.pt")
                torch.save(self.model.state_dict(), model_save_path)
                print(f"保存最佳模型到 {model_save_path}")
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    print(f"早停，最佳F1值: {best_f1:.4f}")
                    break
    
    def load_model(self, model_path):
        """
        加载模型
        :param model_path: 模型路径
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
    
    def predict(self, text):
        """
        预测文本中的实体
        :param text: 输入文本
        :return: 实体列表
        """
        self.model.eval()
        
        # 分词
        tokenized = self.data_processor.tokenizer(
            text,
            max_length=config.max_seq_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 移到设备上
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        token_type_ids = tokenized['token_type_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        
        if self.mode == 'bio':
            # 解码BIO标签
            pred_entities = decode_bio_labels(outputs, self.data_processor.id2label, attention_mask)[0]
        else:
            # 解码GlobalPointer结果
            pred_entities = decode_globalpointer_logits(
                outputs, attention_mask, 
                threshold=config.infer_threshold,
                entity_types=list(self.data_processor.entity_types)
            )[0]
        
        # 转换为原始文本的起止位置
        offset_mapping = self.data_processor.tokenizer(
            text,
            max_length=config.max_seq_len,
            truncation=True,
            padding='max_length',
            return_offsets_mapping=True
        )['offset_mapping']
        
        # 提取实体
        entities = []
        for start, end, label in pred_entities:
            # 获取原始文本的起止位置
            start_char = offset_mapping[start][0]
            end_char = offset_mapping[end][1]
            
            # 提取实体文本
            entity_text = text[start_char:end_char]
            
            # 跳过空实体
            if entity_text:
                entities.append({
                    'label': label,
                    'entity': entity_text,
                    'start_offset': start_char,
                    'end_offset': end_char
                })
        
        return entities


def main():
    """
    主函数
    """
    # 加载数据
    train_path = os.path.join(config.processed_dir, 'train.json')
    val_path = os.path.join(config.processed_dir, 'val.json')
    
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    # 初始化数据处理器
    processor = DataProcessor()
    processor.collect_entity_types(train_data + val_data)
    
    # 转换数据格式
    if config.model_name in ['bert_crf', 'bert_bilstm_crf', 'macbert_bilstm', 'macbert_bilstm_crf', 'macbert_sdip_cnn_mlif_crf', 'macbert_crf']:
        # BIO格式
        train_data = processor.json_to_bio(train_data)
        val_data = processor.json_to_bio(val_data)
    else:
        # GlobalPointer格式
        train_data = processor.json_to_globalpointer(train_data)
        val_data = processor.json_to_globalpointer(val_data)
    
    # 保存标签映射
    processor.save_label_mapping(os.path.join(config.processed_dir, 'label_mapping.json'))
    
    # 初始化训练器
    trainer = ModelTrainer(config.model_name, processor)
    
    # 训练模型
    trainer.train(train_data, val_data)


if __name__ == '__main__':
    main()