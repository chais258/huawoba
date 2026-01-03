import time
import io
import base64
from PIL import ImageGrab
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import dashscope
import os
import dashscope
import pyaudio
import time
import numpy as np
def capture_screen():
    while True:
        # 截屏并保存到内存
        screenshot = ImageGrab.grab()
        img_byte_arr = io.BytesIO()
        screenshot.save(img_byte_arr, format='PNG')
        binary_data = img_byte_arr.getvalue()

        # 转换为Base64编码
        image_data = base64.b64encode(binary_data).decode('utf-8')
        # print(f"屏幕截图的Base64编码: {base64_data[:100]}... (截取前100字符)")
        llm_image=ChatOpenAI(model_name="qwen3-vl-flash", api_key="sk-665b9251561f4ed69e1928b86848702a",base_url=r'https://dashscope.aliyuncs.com/compatible-mode/v1')
        message = HumanMessage(  
            content=[  
                {"type": "text", "text": "你是一只猫娘，正在偷窥主人的屏幕，要对偷窥到的屏幕进行评价，并且用可爱的语气回复我，注意要用赞美的语言，给我提供足够的情绪价值，确保回复内容在200以内"},  
                {  
                    "type": "image_url",  
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},  
                },  
            ],  
        ) 
        response = llm_image.invoke([message])
        print(response.content)
        

        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

        p = pyaudio.PyAudio()
        # 创建音频流
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=24000,
                        output=True)


        text  = response.content
        response2 = dashscope.MultiModalConversation.call(
            api_key="sk-665b9251561f4ed69e1928b86848702a",
            model="qwen3-tts-flash",
            text=text,
            voice="Cherry",
            language_type="Chinese",  # 建议与文本语种一致，以获得正确的发音和自然的语调。
            stream=True
        )
        print(response2)
        for chunk in response2:
            if chunk.output is not None:
                audio = chunk.output.audio
                if audio.data is not None:
                    wav_bytes = base64.b64decode(audio.data)
                    audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
                    # 直接播放音频数据
                    stream.write(audio_np.tobytes())
                if chunk.output.finish_reason == "stop":
                    print("finish at: {} ", chunk.output.audio.expires_at)
            else:
                print("No output in this chunk.")
        time.sleep(0.8)
        # 清理资源
        stream.stop_stream()
        stream.close()
        p.terminate()
        # 等待30秒
        time.sleep(360)

if __name__ == "__main__":
    capture_screen()