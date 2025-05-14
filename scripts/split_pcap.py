from scapy.all import rdpcap, wrpcap
import os

def get_file_size(file_path):
    """获取文件大小（MB）"""
    return os.path.getsize(file_path) / (1024 * 1024)

def split_pcap(input_file, output_dir, num_segments=16, packets_per_segment=100000):
    """
    将PCAP文件分割成指定数量的片段
    
    参数:
    input_file: 输入PCAP文件路径
    output_dir: 输出目录
    num_segments: 片段数量
    packets_per_segment: 每个片段的数据包数量
    """
    try:
        print(f"开始读取文件: {input_file}")
        print(f"文件大小: {get_file_size(input_file):.2f} MB")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 依次生成每个片段
        for segment in range(1, num_segments + 1):
            print(f"\n生成第 {segment} 个片段...")
            
            # 读取数据包
            packets = rdpcap(input_file, count=packets_per_segment)
            if not packets:
                print("没有更多数据包了")
                break
                
            # 保存片段
            output_file = os.path.join(output_dir, f"segment_{segment}.pcap")
            print(f"保存片段 {segment}，共{len(packets)}包")
            wrpcap(output_file, packets)
            print(f"片段 {segment} 大小: {get_file_size(output_file):.2f} MB")
        
        print(f"\n处理完成！共生成 {segment} 个片段")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    input_file = "./data/raw/20240101.pcap"
    output_dir = "./data/split"
    split_pcap(input_file, output_dir) 