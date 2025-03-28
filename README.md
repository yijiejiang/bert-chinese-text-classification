# BERT 中文文本分类项目实践

这个项目实现了基于 BERT 的中文文本分类系统，支持模型训练、导出和推理。项目使用 PyTorch 框架实现，并支持将模型导出为 ONNX 格式以提高推理性能和跨平台兼容性。

数据集采用的是今日头条中文新闻（短文本）分类数据集：[https://github.com/fateleak/toutiao-text-classfication-dataset](https://github.com/fateleak/toutiao-text-classfication-dataset)

## 功能特点

- 基于 BERT 的中文文本分类
- 支持 PyTorch 和 ONNX 两种模型格式
- 自动检测并使用最佳可用设备 (MPS/CUDA/CPU)
- 详细的训练过程和评估指标
- 简单易用的推理接口
- 支持批量处理和单文本预测

## 项目结构

```
bert/
├── data/                  # 数据目录
├── models/                # 预训练模型目录
│   ├── bert-tiny-chinese/ # 预训练的小型中文 BERT 模型
│   │   ├── config.json           # 模型配置文件
│   │   ├── pytorch_model.bin     # 模型权重文件
│   │   ├── special_tokens_map.json # 特殊标记映射
│   │   ├── tokenizer_config.json # 分词器配置
│   │   ├── vocab.txt             # 词汇表
│   │   └── README.md             # 模型说明文档
│   │
│   └── bert-base-chinese/  # 预训练的标准中文 BERT 模型
│       ├── config.json           # 模型配置文件
│       ├── pytorch_model.bin     # 模型权重文件
│       ├── tokenizer.json        # 分词器文件
│       ├── tokenizer_config.json # 分词器配置
│       ├── vocab.txt             # 词汇表
│       └── README.md             # 模型说明文档
│
├── output/                # 输出目录，存放训练好的模型
├── toutiao-text-classfication-dataset/ # 头条文本分类数据集
├── get_data.py            # 数据处理脚本
├── train.py               # 模型训练脚本
├── predict.py             # 模型推理脚本
├── run.sh                 # 运行脚本
└── requirements.txt       # 依赖包列表
```

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：

- torch
- transformers
- numpy
- pandas
- scikit-learn
- tqdm
- onnx
- onnxruntime

## 使用方法

### 1. 数据准备

项目使用头条文本分类数据集，包含15个类别。请提前自行下载数据，数据处理脚本会自动下载并处理数据：

```bash
python get_data.py
```

### 2. 模型训练

准备预训练文件

从 HuggingFace 下载模型文件到 models 文件夹下，项目准备了两个文件ckiplab/bert-tiny-chinese和google-bert/bert-base-chinese，可自行下载文件到指定路径下。

训练脚本会自动检测并使用最佳可用设备（MPS/CUDA/CPU）：

```bash
python train.py
```

训练完成后，模型会被保存到 `output` 目录，包括：

- `final_model.pt`：PyTorch 模型
- `model.onnx`：ONNX 格式模型
- 分词器和配置文件

### 3. 模型推理

#### 使用 PyTorch 模型

```bash
python predict.py \
  --model_type pytorch \
  --model_path output/final_model.pt \
  --text "杂交水稻宣布亩产突破 2000KG"
```

#### 使用 ONNX 模型

```bash
python predict.py \
  --model_type onnx \
  --model_path output/model.onnx \
  --text "杂交水稻宣布亩产突破 2000KG"
```

#### 其他参数

- `--max_length`：设置最大序列长度，默认为 128
- `--num_labels`：设置类别数量，默认为 15

### 4. 快速运行

项目提供了 `run.sh` 脚本，可以快速运行训练和推理：

```bash
# 训练模型
sh run.sh train

# 使用 PyTorch 模型推理
sh run.sh predict_pytorch

# 使用 ONNX 模型推理
sh run.sh predict_onnx
```

## 类别映射

模型支持以下15个类别：

| ID | 类别名称 |
| -- | -------- |
| 0  | 民生     |
| 1  | 文化     |
| 2  | 娱乐     |
| 3  | 体育     |
| 4  | 财经     |
| 5  | 房产     |
| 6  | 汽车     |
| 7  | 教育     |
| 8  | 科技     |
| 9  | 军事     |
| 10 | 旅游     |
| 11 | 国际     |
| 12 | 证券     |
| 13 | 农业     |
| 14 | 电竞     |

## 性能优化

- 使用 ONNX 格式可以显著提高推理速度
- 在 Apple M 系列芯片上使用 MPS 加速
- 支持批量处理以提高吞吐量

## 环境要求

- Python 3.8+
- PyTorch 1.12+
- 对于 GPU 加速：
  - NVIDIA GPU + CUDA（适用于 NVIDIA 显卡）
  - Apple M1/M2/M3 芯片（使用 MPS 加速）

## 许可证

MIT

## 致谢

- 感谢 Hugging Face 提供的 Transformers 库
- 感谢 ONNX 和 ONNX Runtime 团队提供的模型转换和推理工具
