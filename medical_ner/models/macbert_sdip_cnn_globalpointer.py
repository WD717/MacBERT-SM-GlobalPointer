# -*- coding: utf-8 -*-
"""
MacBERT-SDIP-CNN-GlobalPointer模型实现"""
import torch
import torch.nn as nn
from .base_model import BertBaseModel, GlobalPointer, GlobalPointerLoss, SDIP, CNN
from config import config


class MacBERTSDIPCNNGlobalPointer(nn.Module):
    """MacBERT-SDIP-CNN-GlobalPointer模型"""
    
    def __init__(self, num_entities, model_path=None):
        """
        初始化MacBERT-SDIP-CNN-GlobalPointer模型
        :param num_entities: 实体类型数量
        :param model_path: MacBERT模型路径
        """
        super().__init__()
        # MacBERT基础模型
        self.macbert = BertBaseModel(model_path=model_path or config.macbert_model)
        
        # SDIP语义依赖感知组件
        self.sdip = SDIP(
            input_dim=self.macbert.hidden_dim,
            hidden_dim=self.macbert.hidden_dim
        )
        
        # CNN局部特征提取组件
        self.cnn = CNN(
            input_dim=self.macbert.hidden_dim,
            filters=config.cnn_filters,
            kernel_sizes=config.cnn_kernel_sizes
        )
        
        # 融合层，将SDIP输出和CNN输出融合
        self.fusion = nn.Linear(
            self.macbert.hidden_dim + self.cnn.output_dim,
            self.macbert.hidden_dim
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
        
        # SDIP语义依赖增强
        sdip_out = self.sdip(macbert_out)
        
        # CNN局部特征提取
        cnn_out = self.cnn(sdip_out)
        
        # 将CNN输出扩展到序列长度维度
        cnn_out = cnn_out.unsqueeze(1).expand(-1, sdip_out.size(1), -1)
        
        # 融合SDIP输出和CNN输出
        fused = torch.cat([sdip_out, cnn_out], dim=-1)
        fused = self.fusion(fused)
        fused = torch.relu(fused)
        
        # GlobalPointer预测
        logits = self.global_pointer(fused, attention_mask)
        
        if labels is not None:
            # 使用GlobalPointerLoss计算损失
            loss_fn = GlobalPointerLoss()
            loss = loss_fn(logits, labels)
            return loss
        else:
            # 推理模式，返回预测结果
            return logits