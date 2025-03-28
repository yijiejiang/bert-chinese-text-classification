import torch
import argparse
import os
import numpy as np
from transformers import BertTokenizer
from torch.nn import functional as F
import onnxruntime

# 定义分类映射
CLASS_NAMES = {
    0: "民生",
    1: "文化",
    2: "娱乐",
    3: "体育",
    4: "财经",
    5: "房产",
    6: "汽车",
    7: "教育",
    8: "科技",
    9: "军事",
    10: "旅游",
    11: "国际",
    12: "证券",
    13: "农业",
    14: "电竞"
}


def load_pytorch_model(model_path, device, num_labels=15):
    """加载PyTorch模型
    
    参数:
        model_path: 模型路径
        device: 设备
        num_labels: 类别数量
    
    返回:
        model: 加载的模型
        tokenizer: 分词器
    """
    from transformers import BertForSequenceClassification, BertConfig
    
    print(f"正在加载PyTorch模型: {model_path}")
    
    # 加载分词器
    try:
        vocab_file = "models/bert-tiny-chinese/vocab.txt"
        tokenizer = BertTokenizer(vocab_file=vocab_file)
        print(f"分词器从 {vocab_file} 加载成功")
    except Exception as e:
        print(f"加载分词器失败: {e}")
        raise
    
    # 创建模型配置
    config_file = "models/bert-tiny-chinese/config.json"
    config = BertConfig.from_json_file(config_file)
    config.num_labels = num_labels  # 设置类别数量
    print(f"模型配置从 {config_file} 加载成功")
    
    # 创建模型
    model = BertForSequenceClassification(config)
    print("模型创建成功")
    
    # 加载模型权重
    print(f"从 {model_path} 加载模型权重")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("PyTorch模型加载完成")
    return model, tokenizer

def load_onnx_model(model_path):
    """加载ONNX模型
    
    参数:
        model_path: 模型路径
    
    返回:
        session: ONNX会话
        tokenizer: 分词器
    """
    print(f"正在加载ONNX模型: {model_path}")
    
    # 加载分词器
    try:
        vocab_file = "models/bert-tiny-chinese/vocab.txt"
        tokenizer = BertTokenizer(vocab_file=vocab_file)
        print(f"分词器从 {vocab_file} 加载成功")
    except Exception as e:
        print(f"加载分词器失败: {e}")
        raise
    
    # 创建ONNX运行时会话
    session = onnxruntime.InferenceSession(model_path)
    print(f"ONNX会话从 {model_path} 创建成功")
    
    print("ONNX模型加载完成")
    return session, tokenizer

def predict_with_pytorch(text, model, tokenizer, device, max_length=128):
    """使用PyTorch模型进行预测
    
    参数:
        text: 输入文本
        model: PyTorch模型
        tokenizer: 分词器
        device: 设备
        max_length: 最大序列长度
    
    返回:
        predicted_class: 预测的类别
        probabilities: 各类别的概率
    """
    # 对文本进行编码
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # 将编码后的数据移到设备上
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)
    
    # 不计算梯度
    with torch.no_grad():
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取预测结果
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
    
    return predicted_class, probabilities[0].cpu().numpy()

def predict_with_onnx(text, session, tokenizer, max_length=128):
    """使用ONNX模型进行预测
    
    参数:
        text: 输入文本
        session: ONNX会话
        tokenizer: 分词器
        max_length: 最大序列长度
    
    返回:
        predicted_class: 预测的类别
        probabilities: 各类别的概率
    """
    # 对文本进行编码
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # 准备输入
    ort_inputs = {
        "input_ids": encoding['input_ids'].numpy(),
        "attention_mask": encoding['attention_mask'].numpy(),
        "token_type_ids": encoding['token_type_ids'].numpy()
    }
    
    # 运行推理
    logits = session.run(None, ort_inputs)[0]
    
    # 获取预测结果
    probabilities = F.softmax(torch.tensor(logits), dim=1)
    predicted_class = np.argmax(logits, axis=1).item()
    
    return predicted_class, probabilities[0].numpy()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="BERT文本分类预测脚本")
    parser.add_argument("--model_type", type=str, choices=["pytorch", "onnx"], default="pytorch", help="模型类型: pytorch 或 onnx")
    parser.add_argument("--model_path", type=str, default="output/final_model.pt", help="模型路径")
    parser.add_argument("--tokenizer_path", type=str, default="output", help="分词器路径")
    parser.add_argument("--text", type=str, required=True, help="要预测的文本")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--num_labels", type=int, default=15, help="类别数量")
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置设备 - 检测并使用MPS（适用于M1/M2 Mac）
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"使用设备: Apple M系列芯片 (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用设备: NVIDIA GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"使用设备: CPU")
    
    # 根据模型类型加载模型
    if args.model_type == "pytorch":
        model, tokenizer = load_pytorch_model(args.model_path, device, args.num_labels)
        predicted_class, probabilities = predict_with_pytorch(args.text, model, tokenizer, device, args.max_length)
    else:  # onnx
        session, tokenizer = load_onnx_model(args.model_path)
        predicted_class, probabilities = predict_with_onnx(args.text, session, tokenizer, args.max_length)
    
    # 输出预测结果
    print(f"\n输入文本: '{args.text}'")
    print(f"预测类别ID: {predicted_class} ({CLASS_NAMES.get(predicted_class, '未知类别')})")
    
    # 输出各类别概率（带类别名称）
    print("各类别概率:")
    for class_id, prob in enumerate(probabilities):
        class_name = CLASS_NAMES.get(class_id, '未知类别')
        print(f"  {class_id}: {class_name} - {prob:.4f}")
    
    # 输出最高概率的类别
    top_class = np.argmax(probabilities)
    top_prob = probabilities[top_class]
    top_class_name = CLASS_NAMES.get(top_class, '未知类别')
    print(f"\n最高概率类别: {top_class} ({top_class_name}), 概率: {top_prob:.4f}")

if __name__ == "__main__":
    main()
    
