import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import jieba
from tqdm import tqdm

# 定义数据路径
DATA_PATH = "toutiao-text-classfication-dataset/toutiao_cat_data.txt"
OUTPUT_DIR = "data"

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 定义分类映射字典 (分类code -> 数字标签)
category_dict = {
    '100': 0,  # 民生 故事 news_story
    '101': 1,  # 文化 文化 news_culture
    '102': 2,  # 娱乐 娱乐 news_entertainment
    '103': 3,  # 体育 体育 news_sports
    '104': 4,  # 财经 财经 news_finance
    '106': 5,  # 房产 房产 news_house
    '107': 6,  # 汽车 汽车 news_car
    '108': 7,  # 教育 教育 news_edu
    '109': 8,  # 科技 科技 news_tech
    '110': 9,  # 军事 军事 news_military
    '112': 10, # 旅游 旅游 news_travel
    '113': 11, # 国际 国际 news_world
    '114': 12, # 证券 股票 stock
    '115': 13, # 农业 三农 news_agriculture
    '116': 14  # 电竞 游戏 news_game
}

# 分类名称映射 (数字标签 -> 分类名称)
category_names = {
    0: 'news_story',
    1: 'news_culture',
    2: 'news_entertainment',
    3: 'news_sports',
    4: 'news_finance',
    5: 'news_house',
    6: 'news_car',
    7: 'news_edu',
    8: 'news_tech',
    9: 'news_military',
    10: 'news_travel',
    11: 'news_world',
    12: 'stock',
    13: 'news_agriculture',
    14: 'news_game'
}

def clean_text(text):
    """
    清洗文本：
    1. 去除特殊字符和标点符号
    2. 去除多余的空格
    """
    # 保留中文、英文、数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def segment_text(text):
    """
    使用jieba进行中文分词
    """
    words = jieba.cut(text)
    return ' '.join(words)

def process_data(clean=True, segment=False):
    """
    处理今日头条文本分类数据集
    格式：新闻ID_!_分类code_!_分类名称_!_新闻标题_!_新闻关键词
    
    参数：
    - clean: 是否清洗文本
    - segment: 是否进行分词
    """
    print("开始处理数据...")
    
    # 存储数据的列表
    texts = []
    labels = []
    categories = []
    keywords = []
    
    # 读取数据文件
    line_count = 0
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for _ in f:
            line_count += 1
    
    print(f"总行数: {line_count}")
    
    # 重新打开文件并处理
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=line_count, desc="读取数据"):
            try:
                # 分割每行数据
                parts = line.strip().split('_!_')
                if len(parts) >= 4:
                    category_code = parts[1]
                    category_name = parts[2]
                    title = parts[3]
                    keyword = parts[4] if len(parts) > 4 else ""
                    
                    # 确保分类代码在我们的映射中
                    if category_code in category_dict:
                        # 文本处理
                        if clean:
                            title = clean_text(title)
                        if segment:
                            title = segment_text(title)
                        
                        texts.append(title)
                        labels.append(category_dict[category_code])
                        categories.append(category_name)
                        keywords.append(keyword)
            except Exception as e:
                print(f"处理行时出错: {line[:50]}... - {str(e)}")
                continue
    
    # 创建DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels,
        'category': categories,
        'keywords': keywords
    })
    
    # 打印数据集统计信息
    print(f"\n成功处理数据量: {len(df)}")
    print("\n各类别数据分布:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"{category_names[label]} (标签 {label}): {count} 条 ({count/len(df)*100:.2f}%)")
    
    # 分割数据集 (70% 训练, 15% 验证, 15% 测试)
    print("\n分割数据集...")
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    # 保存数据集
    print("保存数据集...")
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    
    # 保存完整数据集
    df.to_csv(os.path.join(OUTPUT_DIR, 'full_dataset.csv'), index=False)
    
    print("\n数据处理完成!")
    print(f"训练集: {len(train_df)} 条 ({len(train_df)/len(df)*100:.2f}%)")
    print(f"验证集: {len(val_df)} 条 ({len(val_df)/len(df)*100:.2f}%)")
    print(f"测试集: {len(test_df)} 条 ({len(test_df)/len(df)*100:.2f}%)")
    print(f"数据已保存到 {OUTPUT_DIR} 目录")
    
    return train_df, val_df, test_df

def create_sample_data(df, sample_size=1000, output_file='sample_data.csv'):
    """
    创建一个小型样本数据集，用于快速测试
    确保每个类别的数据都有合理的比例
    """
    print(f"\n创建样本数据集 {output_file}...")
    # 确保每个类别都有数据
    sample_df = pd.DataFrame()
    for label in range(15):
        label_df = df[df['label'] == label]
        # 如果该类别的数据少于 sample_size/15，则全部使用
        n_samples = min(len(label_df), sample_size // 15)
        if n_samples > 0:
            sample_df = pd.concat([sample_df, label_df.sample(n_samples, random_state=42)])
    
    # 保存样本数据
    sample_path = os.path.join(OUTPUT_DIR, output_file)
    sample_df.to_csv(sample_path, index=False)
    print(f"样本数据已保存到 {sample_path}，共 {len(sample_df)} 条")
    
    # 打印样本数据集的类别分布
    print("样本数据集类别分布:")
    sample_label_counts = sample_df['label'].value_counts().sort_index()
    for label, count in sample_label_counts.items():
        print(f"{category_names[label]} (标签 {label}): {count} 条 ({count/len(sample_df)*100:.2f}%)")
    
    return sample_df

def create_demo_data():
    """
    创建一个非常小的演示数据集，用于快速验证模型训练流程
    """
    print("\n创建演示数据集...")
    # 每个类别只选择少量样本
    demo_size = 50  # 每类选择的样本数
    
    # 读取完整数据集
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'full_dataset.csv'))
    
    demo_df = pd.DataFrame()
    for label in range(15):
        label_df = df[df['label'] == label]
        n_samples = min(len(label_df), demo_size)
        if n_samples > 0:
            demo_df = pd.concat([demo_df, label_df.sample(n_samples, random_state=42)])
    
    # 分割数据集 (60% 训练, 20% 验证, 20% 测试)
    train_df, temp_df = train_test_split(demo_df, test_size=0.4, random_state=42, stratify=demo_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    # 保存数据集
    demo_dir = os.path.join(OUTPUT_DIR, 'demo')
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)
    
    train_df.to_csv(os.path.join(demo_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(demo_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(demo_dir, 'test.csv'), index=False)
    
    print(f"演示数据集已保存到 {demo_dir} 目录")
    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")
    print(f"测试集: {len(test_df)} 条")
    
    return train_df, val_df, test_df

def update_demo_py():
    """
    更新demo.py文件中的数据加载部分，使其使用我们处理好的数据
    """
    print("\n更新demo.py文件...")
    try:
        demo_file = 'demo.py'
        
        with open(demo_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修改load_data函数
        load_data_replacement = '''def load_data(data_file):
    """加载数据，返回文本和标签"""
    print(f"加载数据: {data_file}")
    df = pd.read_csv(data_file)
    return df['text'].values, df['label'].values'''
        
        # 修改main函数中的数据文件路径
        data_path_replacement = '''    # 加载数据
    texts, labels = load_data("data/train.csv")'''
        
        # 使用正则表达式替换
        import re
        content = re.sub(r'def load_data\([^)]*\):.*?return texts, labels', 
                        load_data_replacement, content, flags=re.DOTALL)
        
        content = re.sub(r'    # 加载数据\n    texts, labels = load_data\([^)]*\)', 
                        data_path_replacement, content)
        
        # 保存修改后的文件
        with open('demo_modified.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"已创建修改后的demo文件: demo_modified.py")
        return True
    except Exception as e:
        print(f"修改demo.py时出错: {e}")
        print("请手动修改demo.py中的数据加载部分")
        return False

def main():
    """
    主函数：处理数据并创建各种数据集
    """
    # 处理数据
    train_df, val_df, test_df = process_data(clean=True, segment=False)
    
    # 创建样本数据用于快速测试
    create_sample_data(train_df, sample_size=3000, output_file='sample_train.csv')
    create_sample_data(val_df, sample_size=600, output_file='sample_val.csv')
    
    # 创建演示数据集
    create_demo_data()
    
    # 更新demo.py文件
    update_demo_py()
    
    print("\n数据处理完成！现在您可以使用以下数据文件进行训练:")
    print("1. 完整数据集: data/train.csv, data/val.csv, data/test.csv")
    print("2. 样本数据集: data/sample_train.csv, data/sample_val.csv")
    print("3. 演示数据集: data/demo/train.csv, data/demo/val.csv, data/demo/test.csv")
    
    print("\n使用方法:")
    print("1. 使用原始demo.py: 修改load_data函数以加载CSV文件")
    print("2. 使用修改后的demo_modified.py: 直接运行即可")
    print("   - 快速测试: python demo_modified.py  # 默认使用data/train.csv")
    print("   - 使用样本数据: 修改demo_modified.py中的数据路径为'data/sample_train.csv'")
    print("   - 使用演示数据: 修改demo_modified.py中的数据路径为'data/demo/train.csv'")

if __name__ == "__main__":
    main()