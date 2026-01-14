# 中文医疗实体识别模型

## 1. 项目介绍

本项目实现了11种中文医疗实体识别模型，覆盖BIO标签序列标注和GlobalPointer跨度识别两类范式，基于中文医疗实体识别数据集开发。

### 支持的模型

| 模型名称 | 范式 | 核心组件 |
|---------|------|---------|
| BERT-CRF | BIO序列标注 | BERT + CRF |
| BERT-BiLSTM-CRF | BIO序列标注 | BERT + BiLSTM + CRF |
| BERT-GlobalPointer | 跨度识别 | BERT + GlobalPointer |
| MacBERT-BiLSTM | BIO序列标注 | MacBERT + BiLSTM |
| MacBERT-BiLSTM-CRF | BIO序列标注 | MacBERT + BiLSTM + CRF |
| MacBERT-GlobalPointer | 跨度识别 | MacBERT + GlobalPointer |
| MacBERT-MLIF-GlobalPointer | 跨度识别 | MacBERT + MLIF + GlobalPointer |
| MacBERT-SDIP-CNN-GlobalPointer | 跨度识别 | MacBERT + SDIP + CNN + GlobalPointer |
| BERT-SDIP-CNN-MLIF-GlobalPointer | 跨度识别 | BERT + SDIP + CNN + MLIF + GlobalPointer |
| MacBERT-SDIP-CNN-MLIF-CRF | BIO序列标注 | MacBERT + SDIP + CNN + MLIF + CRF |
| MacBERT-SDIP-CNN-MLIF-GlobalPointer | 跨度识别 | MacBERT + SDIP + CNN + MLIF + GlobalPointer |

## 2. 环境搭建

### 2.1 安装依赖

```bash
# 进入项目目录
cd c:\Users\Lenovo\Desktop\AAA研究生存档\课程学习\自然语言处理\code\medical_ner

# 安装依赖
pip install -r requirements.txt
```

### 2.2 环境要求

- Python 3.9.13
- PyTorch 2.1.0+
- 支持CPU/GPU训练

## 3. 数据准备

### 3.1 数据集格式

数据集采用JSON格式，每条数据包含text文本和多类型实体标注，示例如下：

```json
[
  {
    "annotations": [
      {
        "label": "药品",
        "start_offset": 9,
        "end_offset": 14,
        "entity": "乌鸡白凤丸"
      }
    ],
    "id": 0,
    "text": "【药品商品名称】 乌鸡白凤丸 【药品名称】 乌鸡白凤丸 ..."
  }
]
```

### 3.2 数据放置

将原始JSON数据文件 `medical_ner_entities.json` 放置在 `data/raw/` 目录下：

```
medical_ner/
└── data/
    └── raw/
        └── medical_ner_entities.json
```

### 3.3 数据划分

运行主程序时会自动划分数据为训练集、验证集和测试集，比例为8:1:1。

## 4. 快速开始

### 4.1 训练模型

```bash
# 训练BERT-CRF模型
python main.py --model bert_crf --train

# 训练MacBERT-GlobalPointer模型
python main.py --model macbert_globalpointer --train
```

### 4.2 评估模型

```bash
# 评估BERT-CRF模型
python main.py --model bert_crf --eval
```

### 4.3 推理文本

```bash
# 使用BERT-CRF模型推理
python main.py --model bert_crf --infer

# 自定义推理文本
python main.py --model bert_crf --infer --text "【药品商品名称】 乌鸡白凤丸 【适应症】 补气养血，调经止带。"
```

## 5. 模型参数说明

### 5.1 配置文件

所有超参数配置在 `config.py` 文件中，主要参数包括：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| model_name | 模型名称 | bert_crf |
| max_seq_len | 最大序列长度 | 256 |
| train_batch_size | 训练批次大小 | 4 |
| eval_batch_size | 评估批次大小 | 8 |
| learning_rate | 学习率 | 2e-5 |
| epochs | 训练轮数 | 10 |
| early_stopping_patience | 早停轮数 | 3 |
| lstm_hidden_dim | LSTM隐藏层维度 | 256 |
| lstm_layers | LSTM层数 | 2 |
| gp_dim | GlobalPointer维度 | 64 |
| gp_threshold | GlobalPointer预测阈值 | 0.5 |

### 5.2 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --model | 模型名称 | bert_crf |
| --train | 训练模型 | False |
| --eval | 评估模型 | False |
| --infer | 推理文本 | False |
| --text | 推理文本内容 | 示例文本 |
| --config | 配置文件路径 | 空 |

## 6. 工程结构

```
medical_ner/
├── README.md                    # 项目说明文档
├── requirements.txt             # 依赖清单
├── main.py                     # 主入口文件
├── config.py                   # 超参数配置
├── data/                       # 数据目录
│   ├── raw/                    # 原始数据
│   │   └── medical_ner_entities.json  # 原始JSON数据
│   ├── processed/              # 处理后的数据
│   │   ├── train.json          # 训练集
│   │   ├── val.json            # 验证集
│   │   ├── test.json           # 测试集
│   │   └── label_mapping.json  # 标签映射
│   └── split_data.py           # 数据划分脚本
├── models/                     # 模型定义
│   ├── base_model.py           # 基础组件
│   ├── bert_crf.py             # BERT-CRF模型
│   ├── bert_bilstm_crf.py      # BERT-BiLSTM-CRF模型
│   ├── bert_globalpointer.py   # BERT-GlobalPointer模型
│   ├── macbert_bilstm.py       # MacBERT-BiLSTM模型
│   ├── macbert_bilstm_crf.py   # MacBERT-BiLSTM-CRF模型
│   ├── macbert_globalpointer.py # MacBERT-GlobalPointer模型
│   ├── macbert_mlif_globalpointer.py # MacBERT-MLIF-GlobalPointer模型
│   ├── macbert_sdip_cnn_globalpointer.py # MacBERT-SDIP-CNN-GlobalPointer模型
│   ├── bert_sdip_cnn_mlif_globalpointer.py # BERT-SDIP-CNN-MLIF-GlobalPointer模型
│   ├── macbert_sdip_cnn_mlif_crf.py # MacBERT-SDIP-CNN-MLIF-CRF模型
│   └── macbert_sdip_cnn_mlif_globalpointer.py # MacBERT-SDIP-CNN-MLIF-GlobalPointer模型
├── utils/                      # 工具函数
│   ├── data_processor.py       # 数据处理
│   └── metrics.py              # 评估指标
├── train.py                    # 训练脚本
├── evaluate.py                 # 评估脚本
├── infer.py                    # 推理脚本
└── output/                     # 输出目录
    ├── models/                 # 模型保存
    └── logs/                   # 日志
```

## 7. 评估指标

### 7.1 实体级别指标

- **精确率（Precision）**：预测正确的实体数量 / 预测的实体总数
- **召回率（Recall）**：预测正确的实体数量 / 真实的实体总数
- **F1值（F1-Score）**：2 * 精确率 * 召回率 / (精确率 + 召回率)

### 7.2 指标计算方式

- **微平均（Micro）**：将所有实体类型的TP、FP、FN累加后计算
- **宏平均（Macro）**：计算每个实体类型的指标后取平均值

### 7.3 输出格式

```
实体识别评估指标
==================================================
药品:
  精确率: 0.9231
  召回率: 0.8947
  F1值: 0.9087
  TP: 180, FP: 15, FN: 21
--------------------------------------------------
疾病:
  精确率: 0.8857
  召回率: 0.8571
  F1值: 0.8711
  TP: 120, FP: 15, FN: 20
--------------------------------------------------
微平均:
  精确率: 0.9070
  召回率: 0.8810
  F1值: 0.8938
--------------------------------------------------
宏平均:
  精确率: 0.9044
  召回率: 0.8759
  F1值: 0.8899
==================================================
```

## 8. 技术细节

### 8.1 数据处理

- **BIO标签转换**：将JSON格式的实体标注转换为BIO标签序列
- **GlobalPointer跨度转换**：将实体标注转换为(start, end, label)跨度格式
- **自动实体类型识别**：无需硬编码实体类型，自动从数据中收集

### 8.2 模型实现

- **BERT/MacBERT**：使用HuggingFace Transformers库
- **BiLSTM**：可配置隐藏层维度、层数和dropout
- **CRF**：实现标签转移矩阵约束，支持BIO标签解码
- **GlobalPointer**：支持多实体类型并行预测
- **MLIF**：多粒度特征融合，结合字、词、短语特征
- **SDIP**：语义依赖感知，提取依存句法特征
- **CNN**：3/4/5窗口大小的卷积核，提取局部特征

### 8.3 训练优化

- **早停机制**：当验证集F1值连续下降时自动停止
- **梯度裁剪**：防止梯度爆炸
- **学习率调度**：线性学习率衰减
- **权重衰减**：防止过拟合
