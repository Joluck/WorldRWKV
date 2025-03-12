import gradio as gr
import os
from datetime import datetime
import librosa
import numpy as np

from infer.worldmodel import Worldinfer

llm_path='/home/rwkv/model/rwkv7-0.4b-cnasr-step2'
encoder_path='/home/rwkv/model/facebookhubert-large-ls960-ft'
encoder_type='speech'

# 初始化模型
model = Worldinfer(model_path=llm_path, encoder_type=encoder_type, encoder_path=encoder_path)

# 全局变量存储对话状态
current_state = None

def process_audio(audio, chat_history):
    global current_state
    
    # 检查 audio 是否为 None
    if audio is None:
        return chat_history, None
    
    # 解包元组
    sample_rate, audio_data = audio
    
    # 检查并转换音频数据为浮点数格式
    if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
        # 如果音频数据是整数格式，转换为浮点数并归一化
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    
    # 重采样到 16000 Hz
    resampled_audio = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
    
    # 构造提示文本
    prompt = '\x16Assistant:'
    
    try:
        # 判断是否是第一轮对话
        if current_state is None:
            
            
            # 第一轮对话，state为None
            response, new_state = model.generate(prompt, resampled_audio)
        else:
            # 后续对话，传入上一轮的state
            
            
            response, new_state = model.generate(prompt, resampled_audio, state=None)
        
        # 更新状态
        current_state = new_state
        
        # 更新对话历史
        chat_history = chat_history + [("用户", "【语音输入】"), ("助手", response)]
        
    except Exception as e:
        error_message = f"处理音频时出错: {str(e)}"
        chat_history = chat_history + [("用户", "【语音输入】"), ("助手", error_message)]
        current_state = None  # 出错时重置状态
    
    # 返回更新后的对话历史和None（清空音频输入）
    return chat_history, None

def reset_conversation():
    global current_state
    current_state = None
    return [], None

def clear_audio():
    # 只清除音频输入，不影响对话历史
    return None

# 创建Gradio界面
with gr.Blocks(title="RWKVWORLD-AUDIO") as demo:
    gr.Markdown("# RWKVWORLD-AUDIO")
    gr.Markdown("点击录音按钮开始录音，录音结束后点击'提交录音'进行处理。支持多轮对话，系统会记住对话上下文。")
    
    # 聊天历史
    chatbot = gr.Chatbot(
        label="对话历史",
        height=400
    )
    
    # 音频输入
    with gr.Row():
        audio_input = gr.Audio(
            type="numpy",
            label="语音输入",
            sources=["microphone"]
        )
    
    # 按钮区域
    with gr.Row():
        submit_btn = gr.Button("提交录音", variant="primary")
        new_recording_btn = gr.Button("新录音")
        reset_btn = gr.Button("重置对话")
    
    # 事件绑定
    # 提交录音按钮
    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, chatbot],
        outputs=[chatbot, audio_input]
    )
    
    # 新录音按钮 - 只清除音频输入
    new_recording_btn.click(
        fn=clear_audio,
        inputs=None,
        outputs=[audio_input]
    )
    
    # 重置对话按钮
    reset_btn.click(
        fn=reset_conversation,
        inputs=None,
        outputs=[chatbot, audio_input]
    )

# 启动应用
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)