# -*- coding: utf-8 -*-
"""
MacBERT-MLIF-GlobalPointer模型实现"""
import torch
import torch.nn as nn
from .base_model import BertBaseModel, GlobalPointer, GlobalPointerLoss, MLIF
from config import config


class MacBERTMLIFGlobalPointer(nn.Module):
    """MacBERT-MLIF-GlobalPointer模型"""
    
    def __init__(self, num_entities, model_path=None):
        """
        初始化MacBERT-MLIF-GlobalPointer模型
        :param num_entities: 实体类型数量
        :param model_path: MacBERT模型路径
        """
        super().__init__()
        # MacBERT基础模型
        self.macbert = BertBaseModel(model_path=model_path or config.macbert_model)
        
        # MLIF多粒度融合组件
        self.mlif = MLIF(
            input_dim=self.macbert.hidden_dim,
            hidden_dim=self.macbert.hidden_dim
        )
        
        # GlobalPointer层
        self.global_pointer = GlobalPointer(
            input_dim=self.macbert.hidden_dim,
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
        # MacBERT编码
        macbert_out = self.macbert(input_ids, attention_mask, token_type_ids)
        
        # 多粒度融合（这里简化处理，实际应结合词级嵌入）
        # 假设word_emb是词级嵌入，这里使用macbert_out作为替代
        word_emb = macbert_out
        mlif_out = self.mlif(macbert_out, word_emb)
        
        # GlobalPointer预测
        logits = self.global_pointer(mlif_out, attention_mask)
        
        if labels is not None:
            # 使用GlobalPointerLoss计算损失
            loss_fn = GlobalPointerLoss()
            loss = loss_fn(logits, labels)
            return loss
        else:
            # 推理模式，返回预测结果
            return logits