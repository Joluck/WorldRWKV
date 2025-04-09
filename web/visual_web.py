import gradio as gr
from infer.worldmodel import Worldinfer
from PIL import Image
import re
# 初始化模型
llm_path = '/home/alic-li/RWKV-v7/world_weights/rwkv-0'
encoder_path = '/home/alic-li/RWKV-v7/siglip2-base-patch16-384/'
encoder_type = 'siglip'

# 全局变量存储当前上传的图片和模型状态
current_image = None
current_state = None 
first_question = False # 存储模型状态
# 是否是第一轮对话
# 初始化模型
model = Worldinfer(model_path=llm_path, encoder_type=encoder_type, encoder_path=encoder_path)

# 处理用户输入的核心逻辑
import html  # 导入html库

import re

# 处理用户输入的核心逻辑
def chat_fn(user_input, chat_history, image=None):
    global current_image, current_state, first_question
    
    # 如果上传了新图片，更新当前图片并重置状态
    if image is not None:
        current_image = image
    
    # 如果没有图片，提示用户上传
    if current_image is None:
        bot_response = "请先上传一张图片！"
        chat_history.append((user_input, bot_response))
        return "", chat_history
    
    # 确保图片是PIL Image格式
    if not isinstance(current_image, Image.Image) and current_image != 'none':
        current_image = Image.fromarray(current_image)
    
    # 构造提示文本
    prompt = f'\x16User: {user_input}\x17Assistant:'
    
    # 生成结果，传入当前状态
    try:
        if first_question:
            result, state = model.generate(prompt, current_image, state=None)
        else:
            result, state = model.generate(prompt, 'none', state=current_state)
        
        first_question = False
        bot_response, current_state = result, state
        
        # 解析<think>和</think>标签
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
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
    return "", chat_history  # 清空输入框，更新聊天记录
# 处理图片上传
def update_image(image):
    global current_image, current_state,first_question
    current_image = image
    current_state = None 
    first_question = True
    # print('1111111111111111111',first_question) # 上传新图片时重置状态
    return "图片已上传成功！可以开始提问了。"

# 清空图片
def clear_image():
    global current_image, current_state
    current_image = None
    current_state = None  # 清空图片时重置状态
    # 返回None给image组件，文本给status组件
    return None, "图片已清除，请上传新图片。"

# 清空历史和图片
def clear_all():
    global current_image, current_state
    current_image = None
    current_state = None  # 清空所有时重置状态
    return [], "", "图片和对话已清空，请重新上传图片。"

# 不使用图片输入的聊天函数
def chat_without_image_update(user_input, chat_history):
    return chat_fn(user_input, chat_history)

# 界面布局组件
with gr.Blocks(title="WORLD RWKV", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# WORLD RWKV")
    gr.Markdown("上传一张图片，然后可以进行多轮提问")
    
    with gr.Row():
        # 左侧图片上传区
        with gr.Column(scale=2):
            image_input = gr.Image(
                type="pil", 
                label="上传图片",
                height=400
            )
            
            # 图片状态和操作
            with gr.Row():
                image_status = gr.Textbox(
                    label="图片状态", 
                    value="请上传图片", 
                    interactive=False
                )
                clear_img_btn = gr.Button("删除图片")
        
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
    # 图片上传事件
    image_input.change(
        fn=update_image,
        inputs=[image_input],
        outputs=[image_status]
    )
    
    # 删除图片按钮事件 - 修复输出顺序，确保类型匹配
    clear_img_btn.click(
        fn=lambda: (None, "图片已清除，请上传新图片。"),  # 使用lambda直接返回正确类型
        inputs=None,
        outputs=[image_input, image_status]
    )
    
    # 发送按钮事件
    submit_btn.click(
        fn=chat_fn,
        inputs=[user_input, chatbot, image_input],
        outputs=[user_input, chatbot]
    )
    
    # 输入框回车事件 - 使用不需要图片参数的函数
    user_input.submit(
        fn=chat_without_image_update,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    )
    
    # 清空按钮事件
    clear_btn.click(
        fn=lambda: ([], "", "图片和对话已清空，请重新上传图片。", None),  # 修复返回值
        inputs=None,
        outputs=[chatbot, user_input, image_status, image_input],
        queue=False
    )

# 启动应用
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)