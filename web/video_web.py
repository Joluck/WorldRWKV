import gradio as gr
from infer.worldmodel import Worldinfer
from PIL import Image
import re
import torch
import random
from PIL import Image
from decord import VideoReader
from decord import cpu
import os

# 初始化模型
llm_path = '/home/rwkv/alic-li/WorldRWKV/rwkv7-0.4b-video-siglip-ocr-base/rwkv-0'
encoder_path = 'google/siglip2-base-patch16-384'
encoder_type = 'siglip'

enable_think = False
# 全局变量存储当前上传的视频关键帧和模型状态
current_video_frames = None  # 存储关键帧列表
current_state = None 
first_question = False 
# 是否是第一轮对话
# 初始化模型
model = Worldinfer(model_path=llm_path, encoder_type=encoder_type, encoder_path=encoder_path)

# 处理用户输入的核心逻辑
import html  # 导入html库


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
def chat_fn(user_input, chat_history, video=None):
    global current_video_frames, current_state, first_question

    # 如果上传了新视频，更新当前视频帧并重置状态
    if video is not None:
        current_video_frames = frame_att_generator(video)

    # 如果没有视频帧，提示用户上传
    if current_video_frames is None or len(current_video_frames) == 0:
        bot_response = "请先上传一个视频！"
        chat_history.append((user_input, bot_response))
        return "", chat_history

    # 构造提示文本
    prompt = f'\x16User: {user_input}\x17Assistant:'

    # 生成结果，传入当前状态
    try:
        if first_question:
            result, state = model.generate(prompt, current_video_frames[0], state=None)  # 使用第一帧作为初始图像
        else:
            result, state = model.generate(prompt, 'none', state=current_state)

        first_question = False
        bot_response, current_state = result, state
        if enable_think == True:
            # 解析</think>标签
            think_pattern = re.compile(r'</think>', re.DOTALL)
            think_matches = think_pattern.findall(bot_response)

            # 解析<answer></answer>标签
            answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
            answer_matches = answer_pattern.findall(bot_response)

            # 构造最终的输出
            final_response = ""
            for match in think_matches:
                final_response += f"<details><summary>Think 🤔 </summary>{html.escape(match)}</details>"

            for match in answer_matches:
                final_response += "Answer 💡"
                final_response += "\n"
                final_response += html.escape(match)

            # 转义HTML标签
            bot_response = final_response

    except Exception as e:
        bot_response = f"生成回复时出错: {str(e)}"
        current_state = None  # 出错时重置状态

    # 更新对话历史
    chat_history.append((user_input, bot_response))

    # 返回更新后的组件状态
    return "", chat_history  # 清空输入框，更新聊天记录# 处理图片上传
def update_video(video_path):
    global current_video_frames, current_state, first_question
    if video_path is not None:
        current_video_frames = frame_att_generator(video_path)  # 提取关键帧
        current_state = None
        first_question = True
        return "视频已上传成功！可以开始提问了。"
    else:
        return "视频上传失败，请重新上传。"

# 清空图片
def clear_image():
    global current_state, current_video_frames
    current_video_frames = None  
    current_state = None 
    return None, "图片已清除，请上传新图片。"

# 清空历史和图片
def clear_all():
    global current_video_frames, current_state
    current_video_frames = None
    current_state = None
    return [], "", "图片和对话已清空，请重新上传图片。"

def chat_without_video_update(user_input, chat_history):
    return chat_fn(user_input, chat_history)

# 界面布局组件
with gr.Blocks(title="WORLD RWKV", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# WORLD RWKV")
    gr.Markdown("上传一个视频，然后可以进行多轮提问")

    with gr.Row():
        # 左侧视频上传区
        with gr.Column(scale=2):
            video_input = gr.Video(
                label="上传视频",
                height=400
            )

            # 视频状态和操作
            with gr.Row():
                video_status = gr.Textbox(
                    label="视频状态", 
                    value="请上传视频", 
                    interactive=False
                )
                clear_video_btn = gr.Button("删除视频")

        # 右侧对话区
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="对话记录",
                bubble_full_width=False,
                height=500
            )

    # 控制区域
    with gr.Row():
        # 输入组件
        user_input = gr.Textbox(
            placeholder="请输入问题...",
            scale=7,
            container=False,
            label="问题输入"
        )

        # 操作按钮
        with gr.Column(scale=1):
            submit_btn = gr.Button("发送", variant="primary")
            clear_btn = gr.Button("清空所有")
    # 事件绑定
    # 视频上传事件
    video_input.change(
        fn=update_video,
        inputs=[video_input],
        outputs=[video_status]
    )

    # 删除视频按钮事件
    clear_video_btn.click(
        fn=lambda: (None, "视频已清除，请上传新视频。"),  # 使用lambda直接返回正确类型
        inputs=None,
        outputs=[video_input, video_status]
    )

    # 发送按钮事件
    submit_btn.click(
        fn=chat_fn,
        inputs=[user_input, chatbot, video_input],
        outputs=[user_input, chatbot]
    )

    # 输入框回车事件 - 使用不需要视频参数的函数
    user_input.submit(
        fn=chat_without_video_update,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    )
    
    # 清空按钮事件
    clear_btn.click(
        fn=lambda: ([], "", "图片和对话已清空，请重新上传图片。", None),  # 修复返回值
        inputs=None,
        outputs=[chatbot, user_input, video_status, video_input],
        queue=False
    )

# 启动应用
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)