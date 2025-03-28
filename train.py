import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW  # 使用PyTorch的AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import random
from tqdm import tqdm, trange
import time
import argparse
# 导入混合精度训练所需的库
from torch.cuda.amp import autocast, GradScaler
# 导入ONNX相关库
import onnx
import onnxruntime

# 设置随机种子，确保结果可复现
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# 定义数据集类
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(text, 
                                  add_special_tokens=True,
                                  max_length=self.max_length,
                                  return_token_type_ids=True,
                                  padding='max_length',
                                  truncation=True,
                                  return_attention_mask=True,
                                  return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }



# 假设您已经有了数据，格式为 DataFrame，包含 'text' 和 'label' 列
# 这里我们创建一个示例数据加载函数
def load_data(data_file):
    """加载数据，返回文本和标签"""
    print(f"加载数据: {data_file}")
    df = pd.read_csv(data_file)
    return df['text'].values, df['label'].values

# 训练函数
def train(model, tokenizer, OUTPUT_DIR, train_dataloader, val_dataloader, epochs, optimizer, scheduler, device, use_amp=False):
    """训练模型
    
    参数:
        model: 要训练的模型
        tokenizer: 分词器
        OUTPUT_DIR: 输出目录
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        epochs: 训练轮数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 训练设备
        use_amp: 是否使用混合精度训练
    """
    # 设置模型为训练模式
    model.train()
    
    # 跟踪最佳验证准确率
    best_accuracy = 0
    
    # 记录训练开始时间
    training_start_time = time.time()
    
    # 如果使用混合精度训练，初始化梯度缩放器
    scaler = GradScaler() if use_amp else None
    
    # 创建epoch进度条
    epoch_iterator = trange(epochs, desc="训练进度")
    
    # 训练循环
    for epoch in epoch_iterator:
        epoch_start_time = time.time()
        
        # 初始化每个epoch的总损失和正确预测数
        total_loss = 0
        batch_count = 0
        
        # 创建batch进度条
        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
            position=0
        )
        
        # 训练步骤
        for batch in batch_iterator:
            batch_count += 1
            
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 清除之前的梯度
            model.zero_grad()
            
            # 混合精度训练
            if use_amp:
                with autocast():
                    # 前向传播
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels
                    )
                    loss = outputs.loss
                
                # 使用梯度缩放器进行反向传播
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
            else:
                # 正常精度训练
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新参数
                optimizer.step()
            
            # 更新学习率
            scheduler.step()
            
            # 记录总损失
            total_loss += loss.item()
            
            # 更新进度条信息
            current_lr = scheduler.get_last_lr()[0]
            avg_loss = total_loss / batch_count
            batch_iterator.set_postfix({
                '损失': f'{loss.item():.4f}',
                '平均损失': f'{avg_loss:.4f}',
                '学习率': f'{current_lr:.2e}'
            })
        
        # 计算平均损失
        avg_train_loss = total_loss / len(train_dataloader)
        
        # 评估模型
        val_accuracy, val_report = evaluate(model, val_dataloader, device)
        
        # 计算本轮epoch的耗时
        epoch_time = time.time() - epoch_start_time
        
        # 更新epoch进度条信息
        epoch_iterator.set_postfix({
            '训练损失': f'{avg_train_loss:.4f}',
            '验证准确率': f'{val_accuracy:.4f}',
            '耗时': f'{epoch_time:.1f}秒'
        })
        
        # 打印详细信息
        print(f"\nEpoch {epoch + 1}/{epochs} 完成:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证准确率: {val_accuracy:.4f}")
        print(f"  耗时: {epoch_time:.1f}秒")
        print(f"分类报告:\n{val_report}")
        
        # 如果当前模型是最佳模型，保存它
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            # 保存模型
            model_save_path = os.path.join(OUTPUT_DIR, f"best_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"模型已保存到 {model_save_path}")
            
            # 保存分词器
            tokenizer.save_pretrained(OUTPUT_DIR)
            
    return model

# 评估函数
def evaluate(model, dataloader, device):
    """评估模型"""
    # 设置模型为评估模式
    model.eval()
    
    # 存储预测和真实标签
    predictions = []
    true_labels = []
    
    # 记录评估开始时间
    eval_start_time = time.time()
    
    # 创建评估进度条
    eval_iterator = tqdm(
        dataloader,
        desc="评估模型",
        leave=False,
        position=0
    )
    
    # 不计算梯度
    with torch.no_grad():
        for batch in eval_iterator:
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # 获取预测结果
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            
            # 将预测和真实标签添加到列表中
            batch_predictions = preds.cpu().tolist()
            batch_labels = labels.cpu().tolist()
            predictions.extend(batch_predictions)
            true_labels.extend(batch_labels)
            
            # 计算当前批次的准确率
            batch_accuracy = sum([1 for p, t in zip(batch_predictions, batch_labels) if p == t]) / len(batch_predictions)
            
            # 更新进度条信息
            eval_iterator.set_postfix({
                '批次准确率': f'{batch_accuracy:.4f}',
                '样本数': f'{len(predictions)}'
            })
    
    # 计算总体准确率
    accuracy = accuracy_score(true_labels, predictions)
    
    # 生成分类报告
    report = classification_report(true_labels, predictions)
    
    # 计算评估耗时
    eval_time = time.time() - eval_start_time
    
    # 打印评估耗时
    print(f"  评估耗时: {eval_time:.1f}秒")
    
    return accuracy, report

# 预测函数
def export_to_onnx(model, tokenizer, output_dir, device, max_length=128):
    """将PyTorch模型导出为ONNX格式
    
    参数:
        model: 要导出的模型
        tokenizer: 分词器
        output_dir: 输出目录
        device: 设备
        max_length: 最大序列长度
    """
    print("\n开始导出模型为ONNX格式...")
    
    # 将模型设置为评估模式
    model.eval()
    
    # 将模型复制到CPU上进行导出
    cpu_model = model.to('cpu')
    
    # 创建示例输入
    dummy_text = "这是ONNX导出测试文本"
    encoding = tokenizer(dummy_text,
                         add_special_tokens=True,
                         max_length=max_length,
                         return_token_type_ids=True,
                         padding='max_length',
                         truncation=True,
                         return_attention_mask=True,
                         return_tensors='pt')
    
    # 准备输入 (确保在CPU上)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    token_type_ids = encoding['token_type_ids']
    
    # 设置ONNX模型路径
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    # 定义输入名称
    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    output_names = ["logits"]
    
    # 动态轴定义
    dynamic_axes = {
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "token_type_ids": {0: "batch_size"},
        "logits": {0: "batch_size"}
    }
    
    # 导出模型
    torch.onnx.export(
        cpu_model,                                     # 要导出的模型 (在CPU上)
        (input_ids, attention_mask, token_type_ids),  # 模型输入
        onnx_path,                                    # 输出文件路径
        input_names=input_names,                      # 输入名称
        output_names=output_names,                    # 输出名称
        dynamic_axes=dynamic_axes,                    # 动态轴
        opset_version=14,                            # ONNX操作集版本 (提高到支持scaled_dot_product_attention的版本)
        do_constant_folding=True                      # 是否执行常量折叠优化
    )
    
    # 验证ONNX模型
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX模型验证成功，已保存到 {onnx_path}")
        
        # 测试ONNX模型
        # 创建ONNX运行时会话
        ort_session = onnxruntime.InferenceSession(onnx_path)
        
        # 准备输入
        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "token_type_ids": token_type_ids.numpy()
        }
        
        # 运行推理
        ort_outputs = ort_session.run(None, ort_inputs)
        print("ONNX模型测试成功！")
    except Exception as e:
        print(f"ONNX模型验证失败: {e}")
    
    # 将模型移回原来的设备
    model.to(device)
    
    return onnx_path

def predict_text(text, model, tokenizer, device, max_length=128):
    """预测单个文本的类别"""
    # 设置模型为评估模式
    model.eval()
    
    # 对文本进行编码
    encoding = tokenizer(text, 
                         add_special_tokens=True,
                         max_length=max_length,
                         return_token_type_ids=True,
                         padding='max_length',
                         truncation=True,
                         return_attention_mask=True,
                         return_tensors='pt')
    
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
        _, preds = torch.max(logits, dim=1)
    
    return preds.cpu().item()

# 命令行参数解析函数
def parse_args():
    parser = argparse.ArgumentParser(description="BERT文本分类训练脚本")
    parser.add_argument("--model_path", type=str, default="models/bert-tiny-chinese", help="预训练模型路径")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--data_file", type=str, default="data/train.csv", help="训练数据文件")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--use_amp", action="store_true", help="使用混合精度训练")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()

# 主函数
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)

    # 模型路径
    MODEL_PATH = args.model_path
    OUTPUT_DIR = args.output_dir  # 模型保存路径
    
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
    
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 训练参数
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    MAX_LENGTH = args.max_length
    USE_AMP = args.use_amp
    
    # 打印训练配置
    print(f"训练配置:")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  最大序列长度: {MAX_LENGTH}")
    print(f"  混合精度训练: {USE_AMP}")

    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    # 加载模型并设置分类数为15
    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=15,  # 设置为15个分类
        output_attentions=False,
        output_hidden_states=False,
    )
    
    # 加载数据
    texts, labels = load_data(args.data_file)
    
    # 分割数据集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # 创建数据集
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, max_length=MAX_LENGTH
    )
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, tokenizer, max_length=MAX_LENGTH
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE
    )
    
    # 将模型移到设备上
    model.to(device)
    
    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 设置学习率调度器
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 训练模型
    print(f"\n开始训练{' (混合精度模式)' if USE_AMP else ''}...")
    model = train(model, tokenizer, OUTPUT_DIR, train_dataloader, val_dataloader, EPOCHS, optimizer, scheduler, device, use_amp=USE_AMP)
    
    # 保存最终模型
    model_save_path = os.path.join(OUTPUT_DIR, "final_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"最终模型已保存到 {model_save_path}")
    
    # 导出为ONNX格式
    onnx_path = export_to_onnx(model, tokenizer, OUTPUT_DIR, device, max_length=MAX_LENGTH)
    print(f"ONNX模型已导出到 {onnx_path}")
    
    # 示例预测
    example_text = "这是一个测试文本，用于演示模型预测功能"
    predicted_class = predict_text(example_text, model, tokenizer, device, max_length=MAX_LENGTH)
    print(f"示例文本: '{example_text}'")
    print(f"预测类别: {predicted_class}")

# 如果直接运行此脚本，执行主函数
if __name__ == "__main__":
    main()
