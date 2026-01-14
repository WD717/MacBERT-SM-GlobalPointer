# -*- coding: utf-8 -*-
"""
MacBERT-SDIP-CNN-MLIF-CRF模型实现"""
import torch
import torch.nn as nn
from .base_model import BertBaseModel, BiLSTM, CRF, SDIP, CNN, MLIF
from config import config


class MacBERTSDIPCNNMLIFCRF(nn.Module):
    """MacBERT-SDIP-CNN-MLIF-CRF模型"""
    
    def __init__(self, num_tags, model_path=None):
        """
        初始化MacBERT-SDIP-CNN-MLIF-CRF模型
        :param num_tags: 标签数量
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
        
        # MLIF多粒度融合组件
        self.mlif = MLIF(
            input_dim=self.macbert.hidden_dim,
            hidden_dim=self.macbert.hidden_dim
        )
        
        # 融合层，将CNN输出和MLIF输出融合
        self.fusion = nn.Linear(
            self.macbert.hidden_dim + self.cnn.output_dim,
            self.macbert.hidden_dim
        )
        
        # BiLSTM层
        self.bilstm = BiLSTM(
            input_dim=self.macbert.hidden_dim,
            hidden_dim=config.lstm_hidden_dim,
            num_layers=config.lstm_layers,
            dropout=config.lstm_dropout
        )
        
        # 线性层，将BiLSTM输出映射到标签空间
        self.classifier = nn.Linear(self.bilstm.output_dim, num_tags)
        
        # CRF层
        self.crf = CRF(num_labels=num_tags)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        前向传播
        :param input_ids: 输入token ID
        :param attention_mask: 注意力掩码
        :param token_type_ids: token类型ID
        :param labels: 真实标签
        :return: 损失或解码结果
        """
        # MacBERT编码（字符级）
        char_emb = self.macbert(input_ids, attention_mask, token_type_ids)
        
        # 词级嵌入（这里简化处理，直接使用字符级嵌入作为词级嵌入）
        word_emb = char_emb
        
        # SDIP语义依赖增强
        sdip_out = self.sdip(char_emb)
        
        # CNN局部特征提取
        cnn_out = self.cnn(sdip_out)
        
        # 多粒度融合
        mlif_out = self.mlif(sdip_out, word_emb)
        
        # 将CNN输出扩展到序列长度维度
        cnn_out = cnn_out.unsqueeze(1).expand(-1, mlif_out.size(1), -1)
        
        # 融合MLIF输出和CNN输出
        fused = torch.cat([mlif_out, cnn_out], dim=-1)
        fused = self.fusion(fused)
        fused = torch.relu(fused)
        
        # BiLSTM编码
        lstm_out = self.bilstm(fused, attention_mask)
        
        # 映射到标签空间
        emissions = self.classifier(lstm_out)
        
        if labels is not None:
            # 训练模式，计算损失
            # 处理-100标签：将-100替换为0，同时确保这些位置被mask正确忽略
            labels_clone = labels.clone()
            labels_clone[labels_clone == -100] = 0
            
            # 处理掩码：确保掩码是布尔类型
            mask = attention_mask.bool() if attention_mask is not None else None
            
            # 计算CRF损失
            log_likelihood = self.crf(emissions, labels_clone.long(), mask)
            # 返回负对数似然作为损失
            return -log_likelihood.mean()
        else:
            # 推理模式，解码
            # 处理掩码：确保掩码是布尔类型
            mask = attention_mask.bool() if attention_mask is not None else None
            decoded_seqs = self.crf.viterbi_decode(emissions, mask)
            
            # 将解码结果转换为与输入相同长度的张量
            batch_size, max_len = emissions.shape[:2]
            output = torch.zeros((batch_size, max_len), dtype=torch.long, device=emissions.device)
            
            for i, seq in enumerate(decoded_seqs):
                # 填充实际解码的标签，不足部分用0填充
                seq_len = len(seq)
                output[i, :seq_len] = torch.tensor(seq, dtype=torch.long, device=emissions.device)
            
            return output