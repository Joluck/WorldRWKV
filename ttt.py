import pyaudio
import wave

def record_audio(filename="output.wav", seconds=5):
    # 设置参数
    CHUNK = 1024                # 每个缓冲区的帧数
    FORMAT = pyaudio.paInt16    # 采样格式
    CHANNELS = 1                # 单声道
    RATE = 44100               # 采样率 (Hz)
    
    # 创建 PyAudio 对象
    p = pyaudio.PyAudio()
    
    print(f"开始录音 {seconds} 秒...")
    
    # 打开音频流
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    # 录制音频
    frames = []
    for i in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("录音结束!")
    
    # 停止并关闭音频流
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # 保存为 WAV 文件
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"音频已保存为: {filename}")

# 使用示例：录制5秒钟
record_audio(seconds=5)