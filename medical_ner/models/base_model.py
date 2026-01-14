# -*- coding: utf-8 -*-
"""
基础模型组件，封装BERT/MacBERT基础功能
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
from TorchCRF import CRF


class BertBaseModel(nn.Module):
    """BERT基础模型"""
    
    def __init__(self, model_path=None, hidden_dim=768, dropout=0.1, model_type='bert'):
        """
        初始化BERT基础模型
        :param model_path: BERT模型路径
        :param hidden_dim: 隐藏层维度
        :param dropout: dropout概率
        :param model_type: 模型类型 ('bert' 或 'macbert')
        """
        super().__init__()
        from transformers import BertModel, BertConfig
        
        # MacBERT使用BertModel类加载，通过模型路径区分
        self.bert = BertModel.from_pretrained(model_path) if model_path else BertModel(BertConfig(hidden_size=hidden_dim))
            
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        前向传播
        :param input_ids: 输入token ID
        :param attention_mask: 注意力掩码
        :param token_type_ids: token类型ID
        :return: BERT输出
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取last_hidden_state
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = self.dropout(last_hidden_state)
        
        return last_hidden_state


class BiLSTM(nn.Module):
    """双向LSTM层"""
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1, bidirectional=True):
        """
        初始化BiLSTM
        :param input_dim: 输入维度
        :param hidden_dim: 隐藏层维度
        :param num_layers: 层数
        :param dropout: dropout概率
        :param bidirectional: 是否双向
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,  # 单层时dropout无效
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        
        # 双向LSTM的输出维度是hidden_dim * 2
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
    
    def forward(self, x, attention_mask=None):
        """
        前向传播
        :param x: 输入张量 (batch_size, seq_len, input_dim)
        :param attention_mask: 注意力掩码 (batch_size, seq_len)
        :return: LSTM输出
        """
        # LSTM不直接使用attention_mask，需要处理成pack_padded_sequence
        # 这里简化处理，直接传递
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        return lstm_out


class CNN(nn.Module):
    """CNN层，用于提取局部特征"""
    
    def __init__(self, input_dim, filters=64, kernel_sizes=[3, 4, 5], dropout=0.1):
        """
        初始化CNN
        :param input_dim: 输入维度
        :param filters: 卷积核数量
        :param kernel_sizes: 卷积核大小列表
        :param dropout: dropout概率
        """
        super().__init__()
        
        # 多个卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,  # 输入通道数
                out_channels=filters,  # 输出通道数
                kernel_size=(k, input_dim)  # 卷积核大小
            )
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = filters * len(kernel_sizes)
    
    def forward(self, x):
        """
        前向传播
        :param x: 输入张量 (batch_size, seq_len, input_dim)
        :return: CNN输出
        """
        # 调整形状为 (batch_size, 1, seq_len, input_dim)
        x = x.unsqueeze(1)
        
        # 应用多个卷积层
        conv_outs = []
        for conv in self.convs:
            # 卷积操作
            conv_out = torch.relu(conv(x))
            # 池化操作 (batch_size, filters, 1, 1) -> (batch_size, filters)
            conv_out = torch.max_pool2d(conv_out, kernel_size=(conv_out.size(2), 1)).squeeze(2).squeeze(2)
            conv_outs.append(conv_out)
        
        # 拼接所有卷积输出
        cnn_out = torch.cat(conv_outs, dim=1)
        cnn_out = self.dropout(cnn_out)
        
        return cnn_out


class GlobalPointer(nn.Module):
    """GlobalPointer层，用于实体跨度预测"""
    
    def __init__(self, input_dim, num_entities, head_dim=64, dropout=0.1, RoPE=True):
        """
        初始化GlobalPointer
        :param input_dim: 输入维度
        :param num_entities: 实体类型数量
        :param head_dim: 头维度
        :param dropout: dropout概率
        :param RoPE: 是否使用旋转位置编码
        """
        super().__init__()
        self.num_entities = num_entities
        self.head_dim = head_dim
        self.input_dim = input_dim
        self.RoPE = RoPE
        
        # 单个线性层生成查询和键，输出维度为 num_entities * head_dim * 2
        self.dense = nn.Linear(input_dim, self.num_entities * self.head_dim * 2)
        self.dropout = nn.Dropout(dropout)
    
    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        """
        生成正弦位置编码
        :param batch_size: 批次大小
        :param seq_len: 序列长度
        :param output_dim: 输出维度
        :return: 位置编码 (batch_size, seq_len, output_dim)
        """
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        return embeddings
    
    def forward(self, x, attention_mask=None):
        """
        前向传播
        :param x: 输入张量 (batch_size, seq_len, input_dim)
        :param attention_mask: 注意力掩码 (batch_size, seq_len)
        :return: GlobalPointer输出 (batch_size, num_entities, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # 应用dropout
        x = self.dropout(x)
        
        # 线性变换，生成查询和键
        # 输出形状: (batch_size, seq_len, num_entities * head_dim * 2)
        outputs = self.dense(x)
        
        # 分割为查询和键
        # 形状: (batch_size, seq_len, num_entities, head_dim * 2)
        outputs = outputs.reshape(batch_size, seq_len, self.num_entities, self.head_dim * 2)
        
        # 分离查询和键
        # qw, kw形状: (batch_size, seq_len, num_entities, head_dim)
        qw, kw = outputs[..., :self.head_dim], outputs[..., self.head_dim:]
        
        if self.RoPE:
            # 生成旋转位置编码
            # pos_emb形状: (batch_size, seq_len, head_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.head_dim)
            pos_emb = pos_emb.to(x.device)
            
            # 生成cos和sin位置编码
            # cos_pos, sin_pos形状: (batch_size, seq_len, 1, head_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            
            # 应用旋转编码到查询
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            
            # 应用旋转编码到键
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], dim=-1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        
        # 计算注意力分数
        # s_α(i,j) = (R_i q_{i,α})^T (R_j k_{j,α})
        # 形状: (batch_size, num_entities, seq_len, seq_len)
        logits = torch.einsum('b i e h, b j e h -> b e i j', qw, kw)
        
        # 应用注意力掩码
        if attention_mask is not None:
            # (batch_size, 1, seq_len, seq_len)
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_entities, seq_len, seq_len)
            # 填充位置设为极小值
            logits = logits * pad_mask - (1 - pad_mask) * 1e12
        
        # 排除下三角（i > j），只保留i <= j的情况
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        
        # 输出缩放，这是关键的归一化步骤
        logits = logits / self.head_dim ** 0.5
        
        return logits


# 直接使用torchcrf的CRF类，不封装


class MLIF(nn.Module):
    """MLIF多粒度融合组件"""
    
    def __init__(self, input_dim, hidden_dim=768, dropout=0.1):
        """
        初始化MLIF
        :param input_dim: 输入维度
        :param hidden_dim: 隐藏层维度
        :param dropout: dropout概率
        """
        super().__init__()
        # 多粒度特征融合层
        self.fusion = nn.Linear(input_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, char_emb, word_emb):
        """
        前向传播
        :param char_emb: 字符级嵌入 (batch_size, seq_len, input_dim)
        :param word_emb: 词级嵌入 (batch_size, seq_len, input_dim)
        :return: 融合后的嵌入
        """
        # 拼接字符级和词级嵌入
        combined = torch.cat([char_emb, word_emb], dim=-1)
        # 融合
        fused = self.fusion(combined)
        fused = torch.relu(fused)
        fused = self.dropout(fused)
        
        return fused


class SDIP(nn.Module):
    """SDIP语义依赖感知组件"""
    
    def __init__(self, input_dim, hidden_dim=768, dropout=0.1):
        """
        初始化SDIP
        :param input_dim: 输入维度
        :param hidden_dim: 隐藏层维度
        :param dropout: dropout概率
        """
        super().__init__()
        # 语义依赖特征提取层
        self.dep_linear = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, dep_relations=None):
        """
        前向传播
        :param x: 输入嵌入 (batch_size, seq_len, input_dim)
        :param dep_relations: 依存关系（可选）
        :return: 增强后的嵌入
        """
        # 简化处理，实际应结合依存句法分析
        dep_emb = self.dep_linear(x)
        dep_emb = torch.relu(dep_emb)
        dep_emb = self.dropout(dep_emb)
        
        # 与原嵌入相加
        enhanced = x + dep_emb
        
        return enhanced


class RoPE(nn.Module):
    """旋转位置编码（Rotary Position Embedding）"""
    
    def __init__(self, dim, max_seq_len=512):
        """
        初始化RoPE
        :param dim: 嵌入维度
        :param max_seq_len: 最大序列长度
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # 生成旋转位置编码
        self.cos, self.sin = self._generate_rope()
    
    def _generate_rope(self):
        """
        生成旋转位置编码的cos和sin值
        :return: cos和sin张量，形状为 (max_seq_len, dim)
        """
        # 计算旋转角度
        theta = 1.0 / (10000 ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        
        # 生成位置索引
        position = torch.arange(0, self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        
        # 计算旋转角度矩阵
        angle = position * theta
        
        # 扩展到完整维度
        cos = torch.cos(angle).repeat_interleave(2, dim=-1)
        sin = torch.sin(angle).repeat_interleave(2, dim=-1)
        
        return cos, sin
    
    def forward(self, x):
        """
        应用旋转位置编码
        :param x: 输入张量，形状为 (batch_size, seq_len, dim)
        :return: 应用RoPE后的张量
        """
        batch_size, seq_len, dim = x.shape
        
        # 获取当前序列长度对应的RoPE
        cos = self.cos[:seq_len, :].to(x.device)
        sin = self.sin[:seq_len, :].to(x.device)
        
        # 分离偶数和奇数维度
        x1 = x[..., 0::2]  # (batch_size, seq_len, dim//2)
        x2 = x[..., 1::2]  # (batch_size, seq_len, dim//2)
        
        # 应用旋转
        x_rot = torch.stack([-x2, x1], dim=-1).reshape(batch_size, seq_len, dim)
        x_rope = x * cos.unsqueeze(0) + x_rot * sin.unsqueeze(0)
        
        return x_rope


class GlobalPointerLoss(nn.Module):
    """GlobalPointer类别不平衡损失函数"""
    
    def __init__(self, reduction='mean', pos_weight=1.0):
        """
        初始化损失函数
        :param reduction: 损失聚合方式，'mean'或'sum'
        :param pos_weight: 正样本权重，用于处理类别不平衡
        """
        super().__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight
        # 使用带logits的二元交叉熵损失，更稳定
        self.loss_fn = nn.BCEWithLogitsLoss(
            reduction=reduction, 
            pos_weight=torch.tensor(pos_weight)
        )
    
    def forward(self, logits, labels):
        """
        计算类别不平衡损失
        :param logits: 模型输出，形状为 (batch_size, num_entities, seq_len, seq_len)
        :param labels: 真实标签，形状为 (batch_size, num_entities, seq_len, seq_len)
        :return: 损失值
        """
        # 扁平化处理，BCEWithLogitsLoss期望(batch_size, *)形状
        batch_size, num_entities, seq_len, _ = logits.shape
        
        # 计算正样本数量和负样本数量
        pos_num = torch.sum(labels)
        neg_num = batch_size * num_entities * seq_len * seq_len - pos_num
        
        # 动态调整正负样本权重，处理极度不平衡情况
        # 限制最大权重为10.0，避免模型过于偏向正样本
        if pos_num > 0:
            # 计算正负样本比例
            pos_weight = min(neg_num / (pos_num + 1e-10), 10.0)
            # 负样本权重为1，正样本权重根据样本比例动态计算，但不超过10倍
            weight = torch.where(labels == 1, 
                                torch.tensor(pos_weight).to(labels.device), 
                                torch.tensor(1.0).to(labels.device))
        else:
            weight = torch.tensor(1.0).to(labels.device)
        
        # 计算带权重的二元交叉熵损失
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, 
            labels, 
            weight=weight, 
            reduction='sum' if self.reduction == 'sum' else 'mean'
        )
        
        return loss