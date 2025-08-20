# Transformers Project

这是一个基于PyTorch的Transformer模型实现项目，包含位置编码(Positional Embedding)和文本tokenization功能。

## 项目结构

```
transformers.2025.08.18/
├── transformer.ipynb      # 主要的Transformer模型实现
├── tiktoken_demo.py      # tiktoken使用演示脚本
├── trans_instance.txt    # 文本实例数据
├── sales_textbook.txt    # 销售教科书文本数据
├── requirements.txt      # Python依赖列表
├── environment.yml       # Conda环境配置
└── README.md            # 项目说明文档
```

## 环境要求

- Python 3.9+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速)

## 安装依赖

### 方法1: 使用pip安装

```bash
pip install -r requirements.txt
```

### 方法2: 使用conda环境

```bash
conda env create -f environment.yml
conda activate transformers-project
```

## 使用说明

### 1. 运行tiktoken演示

```bash
python tiktoken_demo.py
```

这个脚本演示了如何使用tiktoken进行文本tokenization，包括：
- 文本编码和解码
- Token与字符的映射关系
- 特殊字符处理

### 2. 使用Jupyter Notebook

```bash
jupyter notebook transformer.ipynb
```

在notebook中包含了：
- Transformer模型的完整实现
- 位置编码(Positional Embedding)机制
- 多头注意力机制
- 前馈神经网络

## 核心功能

### Positional Embedding (位置编码)

位置编码是Transformer模型的关键组件，用于为序列中的每个位置提供位置信息。本项目实现了标准的正弦余弦位置编码。

### Text Tokenization

使用tiktoken库进行文本tokenization，支持：
- 中英文混合文本处理
- 字符级别的token映射
- 特殊字符和标点符号处理

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

MIT License