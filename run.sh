#!/bin/bash

# 训练
# python train.py

# 预测
# 使用PyTorch模型

python predict.py \
--model_type pytorch \
--model_path output/final_model.pt \
--tokenizer_path output \
--text "杂交水稻宣布亩产突破 2000KG"

# 使用ONNX模型

# python predict.py \
# --model_type onnx \
# --model_path output/model.onnx \
# --tokenizer_path output \
# --text "杂交水稻宣布亩产突破 2000KG"

# 使用TensorRT模型
# 此处功能待开发
