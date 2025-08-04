from datasets import load_from_disk, concatenate_datasets, load_dataset
import os
from .utils import pipeline, check_vision_token, process_tokens, bytes_to_audio, process_vision_token, read_and_merge_json, load_jsonl_files, load_vision_text
from tqdm import tqdm


data_file = "/DATA/disk1/step3/xhs8"
subdirs = [os.path.join(data_file, d) for d in os.listdir(data_file) 
        if os.path.isdir(os.path.join(data_file, d))]

# 记录数据集来源信息
dataset_source_map = {}  # idx -> dataset_name
current_idx = 0

if subdirs:
    # 加载每个子目录的数据集
    datasets = []
    for subdir in subdirs:
        dataset = load_from_disk(subdir)
        dataset_name = os.path.basename(subdir)
        
        # 记录这个数据集中每个样本的来源
        for i in range(len(dataset)):
            dataset_source_map[current_idx + i] = dataset_name
        
        datasets.append(dataset)
        current_idx += len(dataset)
        print(f"加载数据集 {dataset_name}: {len(dataset)} 条")
    
    # 连接所有数据集
    data = concatenate_datasets(datasets)
    print(f"已连接{len(datasets)}个子目录的数据集，总大小: {len(data)}")
else:
    # 如果没有子目录，直接加载主目录
    data = load_from_disk(data_file)
    dataset_name = os.path.basename(data_file)
    for i in range(len(data)):
        dataset_source_map[i] = dataset_name
    print(f"从单一目录加载数据集 {dataset_name}，大小: {len(data)}")

for idx in tqdm(range(0, len(data), 1000), desc="Processing samples (every 1000)"):
    try:
        sample = data[idx]
        conversation_text = sample['conversations']
        text_tokens, text_labels = check_vision_token(conversation_text)
        print(f"Successfully processed sample {idx}")
    except Exception as e:
        # 获取数据集来源
        source_dataset = dataset_source_map.get(idx, "Unknown")
        
        print(f"\n{'='*60}")
        print(f"❌ Error processing sample {idx}")
        print(f"📁 Source dataset: {source_dataset}")
        print(f"🐛 Error: {e}")
        print(f"📄 Sample data:")
        try:
            # 安全地打印样本数据，避免过长的输出
            sample_str = str(sample)
            if len(sample_str) > 1000:
                sample_str = sample_str[:1000] + "...(truncated)"
            print(sample_str)
        except:
            print("无法打印样本数据")
        print(f"{'='*60}\n")
        continue

