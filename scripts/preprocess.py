from scapy.all import rdpcap
import pandas as pd
import os

# 分桶函数：将包大小转为分词
def discretize_packet_size(size):
    if size < 100:
        return "small"
    elif size < 500:
        return "medium"
    else:
        return "large"

# 解析PCAP并提取特征
def extract_features(pcap_file):
    print(f"正在处理文件: {pcap_file}")
    packets = rdpcap(pcap_file)
    sequences = []
    labels = []
    
    # 按流（源-目的IP对）分组
    flows = {}
    for pkt in packets:
        if "IP" in pkt:
            flow_key = f"{pkt['IP'].src}:{pkt['IP'].dst}"
            if flow_key not in flows:
                flows[flow_key] = []
            flows[flow_key].append(pkt)
    
    print(f"找到 {len(flows)} 个流")
    
    # 每个流生成一个序列
    for flow_key, pkts in flows.items():
        sizes = [len(pkt) for pkt in pkts[:20]]  # 取前20个包
        sequence = [discretize_packet_size(s) for s in sizes]
        if len(sequence) < 20:
            sequence += ["pad"] * (20 - len(sequence))  # 补齐
        sequences.append(" ".join(sequence))
        
        # 标签：端口443为加密（1），其他为非加密（0）
        port = pkts[0]["IP"].dport if "TCP" in pkts[0] or "UDP" in pkts[0] else 0
        label = 1 if port == 443 else 0
        labels.append(label)
    
    return pd.DataFrame({"sequence": sequences, "label": labels})

# 处理segment1.pcap文件
pcap_file = "./data/segment_1.pcap"
df = extract_features(pcap_file)

print(f"总样本数: {len(df)}")
print(f"加密流量比例: {df['label'].mean():.2%}")

# 保存处理后的数据到data文件夹
df.to_csv("./data/processed_segment1_data.csv", index=False)
print("处理后的数据已保存到 ./data/processed_segment1_data.csv")