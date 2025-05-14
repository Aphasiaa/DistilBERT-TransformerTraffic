import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
import os

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

def validate_model(model, val_loader, device):
    """验证模型性能"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def train_model():
    # 加载数据
    print("加载数据...")
    df = pd.read_csv("./data/processed_segment1_data.csv")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, _ = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # 初始化分词器和模型
    print("初始化模型...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    # 创建数据加载器
    train_dataset = TrafficDataset(train_df, tokenizer)
    val_dataset = TrafficDataset(val_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # 减小batch size
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=1.5e-5)  # 学习率适度的降低，由2e-5降低为1.5e-5
    
    # 训练参数
    val_interval = 100  # 验证间隔不变
    low_loss_count = 0
    processed_samples = 0
    
    print("开始训练...")
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        # 训练一个batch
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 更新已处理样本数
        processed_samples += len(labels)
        
        # 打印训练进度
        print(f"Training: {processed_samples}/{len(train_dataset)} samples, Loss: {loss.item():.4f}")
        
        # 每处理val_interval个样本后验证
        if processed_samples % val_interval == 0:
            val_loss, val_acc = validate_model(model, val_loader, device)
            print(f"\n验证结果 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            
            # 检查早停条件
            if val_loss <= 0.075:  # 放宽早停条件，由0.01改为0.075
                low_loss_count += 1
                if low_loss_count >= 2:
                    print(f"\n达到早停条件！验证loss已连续{low_loss_count}次低于0.075")
                    break
            else:
                low_loss_count = 0
    
    # 保存模型
    print("\n保存模型...")
    os.makedirs("./outputs/model", exist_ok=True)
    model.save_pretrained("./outputs/model")
    tokenizer.save_pretrained("./outputs/model")
    print("模型已保存到 ./outputs/model")

if __name__ == "__main__":
    train_model()