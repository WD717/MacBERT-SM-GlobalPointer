# -*- coding: utf-8 -*-
"""
BERT-GlobalPointer模型实现"""
import torch
import torch.nn as nn
from .base_model import BertBaseModel, GlobalPointer, GlobalPointerLoss
from config import config


class BertGlobalPointer(nn.Module):
    """BERT-GlobalPointer模型"""
    
    def __init__(self, num_entities, model_path=None):
        """
        初始化BERT-GlobalPointer模型
        :param num_entities: 实体类型数量
        :param model_path: BERT模型路径
        """
        super().__init__()
        # BERT基础模型
        self.bert = BertBaseModel(model_path=model_path)
        
        # GlobalPointer层
        self.global_pointer = GlobalPointer(
            input_dim=self.bert.hidden_dim,
            num_entities=num_entities,
            head_dim=config.gp_dim
        )
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        前向传播
        :param input_ids: 输入token ID
        :param attention_mask: 注意力掩码
        :param token_type_ids: token类型ID
        :param labels: 真实标签
        :return: 损失或预测结果
        """
        # BERT编码
        bert_out = self.bert(input_ids, attention_mask, token_type_ids)
        
        # GlobalPointer预测
        logits = self.global_pointer(bert_out, attention_mask)
        
        if labels is not None:
            # 使用GlobalPointerLoss计算损失
            loss_fn = GlobalPointerLoss()
            loss = loss_fn(logits, labels)
            return loss
        else:
            # 推理模式，返回预测结果
            return logits