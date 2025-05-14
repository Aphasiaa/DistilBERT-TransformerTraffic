import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
from scipy.special import softmax

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 自定义数据集
class TrafficDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.sequences = df["sequence"].values
        self.labels = df["label"].values
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def evaluate_model():
    print("加载数据...")
    df = pd.read_csv("./data/processed_segment1_data.csv")
    _, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print("加载模型...")
    model_path = "./outputs/model"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    # 创建测试数据加载器
    test_dataset = TrafficDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("开始评估...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_sequences = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            probs = softmax(outputs.logits.cpu().numpy(), axis=1)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            all_sequences.extend([test_dataset.sequences[i] for i in range(len(labels))])
    
    # 计算平均损失
    avg_loss = total_loss / len(test_loader)
    print(f"\n测试集平均损失: {avg_loss:.4f}")
    
    # 生成分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds))
    
    # 创建test_results目录
    results_dir = "./outputs/test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()
    
    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(all_labels, [p[1] for p in all_probs])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, "roc_curve.png"))
    plt.close()
    
    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(all_labels, [p[1] for p in all_probs])
    average_precision = average_precision_score(all_labels, [p[1] for p in all_probs])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR曲线 (AP = {average_precision:.2f})')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(results_dir, "pr_curve.png"))
    plt.close()
    
    # 绘制置信度分布
    plt.figure(figsize=(10, 6))
    correct_probs = [p[1] for i, p in enumerate(all_probs) if all_preds[i] == all_labels[i]]
    incorrect_probs = [p[1] for i, p in enumerate(all_probs) if all_preds[i] != all_labels[i]]
    
    plt.hist(correct_probs, bins=20, alpha=0.5, label='正确预测')
    plt.hist(incorrect_probs, bins=20, alpha=0.5, label='错误预测')
    plt.xlabel('预测置信度')
    plt.ylabel('样本数量')
    plt.title('预测置信度分布')
    plt.legend()
    plt.savefig(os.path.join(results_dir, "confidence_distribution.png"))
    plt.close()
    
    # 错误分析
    print("\n错误分析:")
    error_indices = [i for i, (pred, label) in enumerate(zip(all_preds, all_labels)) if pred != label]
    print(f"总错误样本数: {len(error_indices)}")
    
    if len(error_indices) > 0:
        print("\n前5个错误预测样本:")
        for i in error_indices[:5]:
            print(f"序列: {all_sequences[i]}")
            print(f"真实标签: {all_labels[i]}")
            print(f"预测标签: {all_preds[i]}")
            print(f"预测概率: {all_probs[i]}")
            print("---")
    
    print(f"\n所有评估结果已保存到 {results_dir} 目录")

if __name__ == "__main__":
    evaluate_model()