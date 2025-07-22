from world.utils import load_jsonl_files
import os

data_file = '/home/rwkv/data/vision_step2'  # 请替换为你的实际数据目录路径
datas = load_jsonl_files(f'{data_file}/text/*.jsonl')

missing_images = []
total_count = 0

for data in datas:
    img_name = data['image']
    image_path = f'{data_file}/{img_name}'
    
    if not os.path.exists(image_path):
        missing_images.append(image_path)
    
    total_count += 1

# 打印缺失的图片路径
print("Missing images:")
for path in missing_images:
    print(path)

# 打印统计信息
print(f"\nTotal images checked: {total_count}")
print(f"Number of missing images: {len(missing_images)}")
print(f"Missing percentage: {len(missing_images)/total_count*100:.2f}%")