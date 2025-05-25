import torch
import random
from PIL import Image
from decord import VideoReader
from decord import cpu
import os

def frame_att_generator(video_path, threshold=0.05, min_k=3, max_k=10):
    vr = VideoReader(video_path, ctx=cpu(0))  # 使用 CPU 解码
    fps = vr.get_avg_fps()  # 获取视频平均帧率
    sampling_interval = int(fps)  # 每秒采样一帧作为基准

    frames = []
    frames_flattened = []

    for idx in range(len(vr)):
        # 只对采样帧进行处理
        if idx % sampling_interval == 0:
            frame = vr[idx].asnumpy()  # 转换为 numpy 数组
            frame_rgb = frame / 255.0  # 归一化到 [0, 1]
            frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1).half()
            flat = frame_tensor.reshape(-1)  # 展平缓存
            frames.append(frame_tensor)
            frames_flattened.append(flat)

    if len(frames) <= 1:
        return frames

    # 批量计算帧差
    flattened_tensor = torch.stack(frames_flattened)  # shape: (N, C*H*W)
    diffs = torch.mean(torch.abs(flattened_tensor[1:] - flattened_tensor[:-1]), dim=1)
    selected_indices_sampled = [0] + [i + 1 for i, diff in enumerate(diffs) if diff > threshold]

    K = len(selected_indices_sampled)

    # 如果帧太少，补充随机帧
    if K < min_k:
        candidates = [i for i in range(len(frames)) if i not in selected_indices_sampled]
        missing = min_k - K
        selected_indices_sampled += random.sample(candidates, missing)
        selected_indices_sampled = sorted(selected_indices_sampled)

    # 如果帧太多，保留前 max_k 个差异最大的帧
    elif K > max_k:
        frame_diffs = [(diff.item(), i + 1) for i, diff in enumerate(diffs)]
        frame_diffs.sort(reverse=True, key=lambda x: x[0])
        top_indices = [0] + [idx for diff, idx in frame_diffs[:max_k - 1]]
        selected_indices_sampled = sorted(top_indices)

    # 返回 PIL.Image.Image 图片列表
    return [
        Image.fromarray((frames[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        for i in selected_indices_sampled
    ]

if __name__ == "__main__":
    import time
    video_path = "/home/alic-li/Videos/VID_20241030121322.mp4"
    output_folder = "./output_frames"
    os.makedirs(output_folder, exist_ok=True)

    start_time = time.time()
    key_frames_pil = frame_att_generator(video_path)
    end_time = time.time()

    total_frames = len(key_frames_pil)
    processing_time = end_time - start_time
    fps = total_frames / processing_time

    print(f"共处理 {total_frames} 帧，耗时 {processing_time:.2f} 秒，处理速度：{fps:.2f} FPS")

    for i, img in enumerate(key_frames_pil):
        img.save(os.path.join(output_folder, f"key_frame_{i}.jpg"))