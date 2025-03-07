import gradio as gr
import os
import librosa
import numpy as np
from PIL import Image

from infer.worldmodel import Worldinfer

# 初始化模型路径
# ASR模型 - 语音识别
asr_llm_path = '/home/rwkv/model/rwkv-0.4b-wavlm-asr-demo'
asr_encoder_path = '/home/rwkv/model/facebookhubert-large-ls960-ft'
asr_encoder_type = 'speech'

# Visual模型 - 图像理解
visual_llm_path = '/home/rwkv/model/rwkv7-3b-siglip/rwkv-0'
visual_encoder_path = '/home/rwkv/model/siglip2basep16s384'
visual_encoder_type = 'siglip'

# 初始化两个模型
asr_model = Worldinfer(model_path=asr_llm_path, encoder_type=asr_encoder_type, encoder_path=asr_encoder_path)
visual_model = Worldinfer(model_path=visual_llm_path, encoder_type=visual_encoder_type, encoder_path=visual_encoder_path)

# 全局变量
current_image = None
visual_state = None
first_question = True

def process_audio_and_image(audio, image, chat_history):
    global current_image, visual_state, first_question
    
    # 检查是否有图片
    if image is not None:
        current_image = image
        visual_state = None
        first_question = True
    
    # 如果没有图片，提示用户上传
    if current_image is None:
        return chat_history + [("系统", "请先上传一张图片！")], None
    
    # 检查是否有音频
    if audio is None:
        return chat_history + [("系统", "未检测到语音输入，请重新录制。")], None
    
    # 处理音频 - 转换为文本
    try:
        # 解包音频数据
        sample_rate, audio_data = audio
        
        # 检查并转换音频数据为浮点数格式
        if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
        
        # 重采样到 16000 Hz
        resampled_audio = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        
        # 使用ASR模型将语音转换为文本
        asr_prompt = '\x16Assistant:'
        transcription, _ = asr_model.generate(asr_prompt, resampled_audio)
        
        # 显示转录结果 - 使用明确的格式
        chat_history = chat_history + [("用户(语音识别)", f"「{transcription}」")]
        
        # 确保图片是PIL Image格式
        if not isinstance(current_image, Image.Image) and current_image != 'none':
            current_image = Image.fromarray(current_image)
        
        # 构造提示文本给Visual模型
        visual_prompt = f'\x16User: {transcription}\x17Assistant:'
        
        # 使用Visual模型回答问题
        if first_question:
            # 第一个问题，传入图片
            result, state = visual_model.generate(visual_prompt, current_image, state=None)
            first_question = False
        else:
            # 后续问题，不传入图片
            result, state = visual_model.generate(visual_prompt, 'none', state=visual_state)
        
        # 更新状态
        visual_state = state
        
        # 添加回复到对话历史
        chat_history = chat_history + [("助手", result)]
        
    except Exception as e:
        error_message = f"处理失败: {str(e)}"
        chat_history = chat_history + [("系统", error_message)]
        visual_state = None
        first_question = True
    
    # 返回更新后的对话历史和None（清空音频输入）
    return chat_history, None

def update_image(image):
    global current_image, visual_state, first_question
    current_image = image
    visual_state = None
    first_question = True
    return "图片已上传成功！可以开始语音提问了。"

def clear_audio():
    # 只清除音频输入，不影响对话历史
    return None

def reset_conversation():
    global current_image, visual_state, first_question
    current_image = None
    visual_state = None
    first_question = True
    return [], None, None, "请上传图片并开始对话"

# 创建Gradio界面
with gr.Blocks(title="WORLDRWKV-MULTIMODAL", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# WORLDRWKV-MULTIMODAL")
    
    
    with gr.Row():
        # 左侧图片上传区
        with gr.Column(scale=2):
            image_input = gr.Image(
                type="pil", 
                label="上传图片",
                height=400
            )
            
            # 图片状态
            image_status = gr.Textbox(
                label="图片状态", 
                value="请上传图片", 
                interactive=False
            )
            
            # 音频输入
            audio_input = gr.Audio(
                type="numpy",
                label="语音输入",
                sources=["microphone"]
            )
        
        # 右侧对话区
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="对话历史",
                bubble_full_width=False,
                height=500
            )
            
            # 添加语音识别结果显示区
            with gr.Accordion("最新语音识别结果", open=False):
                asr_result = gr.Textbox(
                    label="语音识别文本",
                    value="尚未进行语音识别",
                    interactive=False
                )
    
    # 按钮区域
    with gr.Row():
        submit_btn = gr.Button("提交问题", variant="primary")
        new_recording_btn = gr.Button("新录音")
        clear_btn = gr.Button("重置对话")
    
    # 事件绑定
    # 图片上传事件
    image_input.change(
        fn=update_image,
        inputs=[image_input],
        outputs=[image_status]
    )
    
    # 提交问题按钮
    submit_btn.click(
        fn=process_audio_and_image,
        inputs=[audio_input, image_input, chatbot],
        outputs=[chatbot, audio_input]
    )
    
    # 新录音按钮
    new_recording_btn.click(
        fn=clear_audio,
        inputs=None,
        outputs=[audio_input]
    )
    
    # 重置对话按钮
    clear_btn.click(
        fn=reset_conversation,
        inputs=None,
        outputs=[chatbot, audio_input, image_input, image_status]
    )

# 启动应用
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) 