import os
import requests
from PIL import Image
from typing import Dict, Any, List, Tuple
import hydra
import torch
import torchvision
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from config_schema import MainConfig
from google import genai
from google.genai import types
# from google.ai.generativelanguage_v1beta import types
import openai
from openai import OpenAI
import anthropic
import json
import argparse
from utils import (
    get_api_key,
    hash_training_config,
    setup_wandb,
    ensure_dir,
    encode_image,
    get_output_paths,
    encode_image_to_base64
)
import argparse
# Define valid image extensions
VALID_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".JPEG"]


# def setup_gemini(api_key: str):
#     return genai.Client(api_key=api_key)


def setup_claude(api_key: str):
    return OpenAI(
        api_key=api_key,
        base_url="https://www.dmxapi.cn/v1",
    )
  # return anthropic.Anthropic(api_key=api_key)
    
def setup_gemini(api_key: str):
    return genai.Client(
        api_key=api_key,                           # 使用配置的 DMXAPI 密钥
        http_options={'base_url': "https://www.dmxapi.cn"}        # 设置自定义 DMXAPI 端点
    )
    # return genai.Client(api_key=api_key)

def setup_gpt4o(api_key: str):
    return OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=api_key,
        base_url="https://www.dmxapi.cn/v1",
    )

def setup_qwen(api_key: str):
    return OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=api_key,
        base_url="https://www.dmxapi.cn/v1",
    )

def setup_qwen3(api_key: str):
    return OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=api_key,
        base_url="https://www.dmxapi.cn/v1",
    )

def setup_client(api_key: str):
    return OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=api_key,
        base_url="https://www.dmxapi.cn/v1",
    )


def get_media_type(image_path: str) -> str:
    """Get the correct media type based on file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    else:
        raise ValueError(f"Unsupported image extension: {ext}")


class ImageDescriptionGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Get API key for the model
        api_key = get_api_key('gpt4o')
        # self.client = setup_client(api_key)
        if "gemini" in model_name:
            self.client = setup_gemini(api_key)
        else:
            self.client = setup_client(api_key)
        # elif "claude" in model_name:
        #     self.client = setup_claude(api_key)
        # elif "gpt4o" in model_name:
        #     self.client = setup_gpt4o(api_key)
        # elif "qwen" in model_name:
        #     self.client = setup_qwen(api_key)
        # # elif "qwen" in model_name:
        # #     self.client = setup_qwen3(api_key)
        # else:
        #     raise ValueError(f"Unsupported model: {model_name}")

    def generate_description(self, image_path: str, version: str) -> str:
        if "gemini" in self.model_name:
            return self._generate_gemini(image_path, version)
        elif "claude" in self.model_name:
            return self._generate_claude(image_path, version)
        # elif self.model_name == "claude4":
        #     return self._generate_claude(image_path, version)
        elif "gpt-5" in self.model_name:
            return self._generate_gpt5(image_path, version)
        elif "gpt" in self.model_name:
            return self._generate_gpt4o(image_path, version)
        elif "qwen" in self.model_name:
            return self._generate_qwen(image_path, version)
        # elif self.model_name == "qwen3":
        #     return self._generate_qwen(image_path, "qwen3")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_gemini(self, image_path: str, model: str) -> str:
        base64_image = encode_image(image_path)
        try:
            response = self.client.models.generate_content(
                model=model,
                # contents=["Describe this image, no longer than 25 words.", image],
                contents=[
                    types.Content(
                        parts=[
                            types.Part(text="Describe this image, no longer than 25 words."),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/jpeg",
                                    data=base64_image
                                ),
                                media_resolution={"level": "media_resolution_high"}
                            )
                        ]
                    )
                ]
            )
        except Exception as e:
            print(f'resp:{e}')
        return response.text.strip()


    def _generate_gpt5(self, image_path: str, model: str) -> str:
        # print(image_path)
        base64_image = encode_image_to_base64(image_path)
        media_type = get_media_type(image_path)
        payload = {
            "model": model,  # 指定使用的gpt5模型版本
            "input": [
                {
                    # 用户消息：包含文本提示和图像URL
                    "role": "user",
                    "content": [
                        # 文本部分：要求AI分析图像内容
                        {"type": "input_text", "text": "Describe this image in one concise sentence, no longer than 20 words."},
                        # {
                        #     # 图像部分：提供要分析的图像URL
                        #     "type": "input_image",
                        #     "image": {
                        #         "data": base64_image,
                        #         "mime_type": "image/jpeg"
                        #     }
                        # },
                        {"type": "input_image", "image_url": base64_image}
                    ],
                },
            ],
            # "stream": True  # 启用流式输出模式，实时接收响应数据
        }
        
        # ==================== HTTP请求头设置 ====================
        # 配置HTTP请求头，包含认证信息和内容类型
        API_KEY = ""
        url = "https://www.dmxapi.cn/v1/responses"
        headers = {
            "Content-Type": "application/json",           # 指定请求内容类型为 JSON
            "Authorization": f"Bearer {API_KEY}"          # 携带 API 密钥进行身份验证
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
    
            # 检查 HTTP 响应状态码，如果出错会抛出异常
            response.raise_for_status()
            data = response.json()
            return data['output'][1]['content'][0]['text']

        except Exception as e:
            # 处理其他未预期的异常
            print(f"未知错误: {e}")
        # print(f"contents:{contents}")
        return data['output'][1]['content'][0]['text']

    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_claude(self, image_path: str, model: str) -> str:
        # print(image_path)
        base64_image = encode_image(image_path)
        media_type = get_media_type(image_path)
        payload = json.dumps({
            "model": model,  # 指定使用的Claude模型版本
            "messages": [
                {
                    # 用户消息：包含文本提示和图像URL
                    "role": "user",
                    "content": [
                        # 文本部分：要求AI分析图像内容
                        {"type": "text", "text": "Describe this image in one concise sentence, no longer than 20 words."},
                        {
                            # 图像部分：提供要分析的图像URL
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                },
            ],
            "stream": True  # 启用流式输出模式，实时接收响应数据
        })
        
        # ==================== HTTP请求头设置 ====================
        # 配置HTTP请求头，包含认证信息和内容类型
        API_KEY = ""
        url = "https://www.dmxapi.cn/v1/chat/completions"
        headers = {
            "Accept": "application/json",                    # 指定接受JSON格式的响应
            "Authorization": f"Bearer {API_KEY}",           # Bearer token认证方式
            "Content-Type": "application/json",             # 指定请求体为JSON格式
        }
        
        try:
            # ==================== 发送HTTP POST请求 ====================
            # 向Claude API发送POST请求，启用流式传输模式
            response = requests.post(url, headers=headers, data=payload, stream=True)
            response.raise_for_status()  # 检查HTTP状态码，如果有错误会抛出异常
            contents = ""
            # ==================== 流式响应处理 ====================
            # 逐行处理流式响应数据
            for line in response.iter_lines():
                if line:  # 确保当前行不为空
                    try:
                        # 尝试将字节数据解码为UTF-8字符串
                        decoded_line = line.decode('utf-8')
                    except UnicodeDecodeError:
                        # 如果解码失败（可能是编码问题），跳过当前行
                        continue
                        
                    # ==================== Server-Sent Events格式处理 ====================
                    # 检查是否为SSE格式的数据行（以"data: "开头）
                    if decoded_line.startswith("data: "):
                        json_data = decoded_line[6:]  # 移除"data: "前缀，获取纯JSON数据
                        
                        # 检查是否为流式传输结束标志
                        if json_data.strip() == "[DONE]":
                            break  # 结束数据处理循环
                        
                        # 跳过空数据行
                        if not json_data.strip():
                            continue
                        
                        try:
                            # ==================== JSON数据解析 ====================
                            # 将JSON字符串解析为Python字典对象
                            data = json.loads(json_data)                            
                            # ==================== 内容提取和输出 ====================
                            # 从响应数据中提取AI生成的内容
                            if "choices" in data and len(data["choices"]) > 0:
                                # 获取第一个选择项的增量数据
                                delta = data["choices"][0].get("delta", {})
                                # 提取文本内容
                                content = delta.get("content")
                                if content:
                                    # 实时输出内容，不换行，立即刷新缓冲区
                                    # end=""：不自动添加换行符
                                    # flush=True：立即刷新输出缓冲区，确保内容立即显示
                                    contents+=content
                                    print(content, end="", flush=True)
                                    
                        except json.JSONDecodeError:
                            # 静默跳过JSON解析错误
                            # 这通常是由于网络传输中的数据分片导致的临时性问题
                            continue
                        except KeyError:
                            # 静默跳过键错误
                            # 当响应数据结构不符合预期时可能发生
                            continue
        
        # ==================== 异常处理 ====================
        except requests.exceptions.RequestException as e:
            # 处理网络请求相关的异常（连接错误、超时等）
            print(f"请求错误: {e}")
        except KeyboardInterrupt:
            # 处理用户手动中断程序（Ctrl+C）
            print("\n\n用户中断了流式输出")
        except Exception as e:
            # 处理其他未预期的异常
            print(f"未知错误: {e}")
        # print(f"contents:{contents}")
        return contents

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_gpt4o(self, image_path: str, model: str) -> str:
        base64_image = encode_image(image_path)
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        # print(f'gpt5o:{response}')
        return response.choices[0].message.content

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_qwen(self, image_path: str, model: str) -> str:
        base64_image = encode_image(image_path)
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        return response.choices[0].message.content

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_qwen3(self, image_path: str, model: str) -> str:
        base64_image = encode_image(image_path)
        response = self.client.chat.completions.create(
            model="qwen3-vl-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        return response.choices[0].message.content


def save_descriptions(descriptions: List[Tuple[str, str]], output_file: str):
    """Save image descriptions to file."""
    ensure_dir(os.path.dirname(output_file))
    with open(output_file, "w", encoding="utf-8") as f:
        # for desc in descriptions:
        #     f.write(f"{desc}\n")
        for filename, desc in descriptions:
            f.write(f"{filename}: {desc}\n")


# @hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main():
    # Initialize wandb using shared utility
    # setup_wandb(cfg)
    args = parser.parse_args()
    # Get config hash and setup paths
    # config_hash = hash_training_config(cfg)
    model_names = args.model_names #["qwen3"] # "qwen", "gpt4o", "claude"
    method = args.method #"foattack"
    for model_name in model_names:
        print(f"Using training output for config hash: {model_name}")
        paths = {}
        paths["output_dir"] = f"/root/autodl-tmp/codes/M-Attack/LAT/img/{method}"
        paths["desc_output_dir"] = f"/root/autodl-tmp/codes/M-Attack/LAT/description/{method}"
        paths["tgt_desc_output_dir"] = f"/root/autodl-tmp/codes/M-Attack/LAT/description"
        tgt_data_path = "./resources/images/target_images"
        # paths = get_output_paths(cfg, config_hash)
        ensure_dir(paths["desc_output_dir"])
    
        try:
            # Initialize description generator
            generator = ImageDescriptionGenerator(model_name=model_name)
    
            # Process original and adversarial images
            tgt_descriptions = []
            adv_descriptions = []
    
            # Walk through the output directory for adversarial images
            print("Processing images...")
            for root, _, files in os.walk(paths["output_dir"]):
                files = sorted(files)
                for file in tqdm(files):
                    # Check if file has valid image extension
                    if any(
                        file.lower().endswith(ext.lower()) for ext in VALID_IMAGE_EXTENSIONS
                    ):
                        try:
                            # Get paths
                            adv_path = os.path.join(root, file)
                            # Extract just the filename without extension
                            filename_base = os.path.splitext(os.path.basename(adv_path))[0]
    
                            # Try each valid extension for target image
                            target_found = False
                            for ext in VALID_IMAGE_EXTENSIONS:
                                tgt_path = os.path.join(
                                    tgt_data_path, "1", filename_base + ext
                                )
                                if os.path.exists(tgt_path):
                                    target_found = True
                                    break
    
                            if target_found:
                                # Generate descriptions
                                if os.path.exists(os.path.join(
                                    paths["tgt_desc_output_dir"], f"target_{model_name}.txt"
                                ))==False:
                                    tgt_desc = generator.generate_description(tgt_path, model_name)
                                    tgt_descriptions.append((file, tgt_desc))
                                adv_desc = generator.generate_description(adv_path, model_name)
                                adv_descriptions.append((file, adv_desc))
                                # print(f'adv:{adv_desc}')
                                # Log to wandb
                                # wandb.log(
                                #     {
                                #         f"descriptions/{file}/target": tgt_desc,
                                #         f"descriptions/{file}/adversarial": adv_desc,
                                #     }
                                # )
    
                            else:
                                print(
                                    f"Target image not found for {filename_base} with any valid extension, skip it."
                                )
    
                        except Exception as e:
                            print(f"Error processing {file}: {e}")
    
            # Save descriptions
            if os.path.exists(os.path.join(
                paths["tgt_desc_output_dir"], f"target_{model_name}.txt"
            ))==False:
                save_descriptions(
                    tgt_descriptions,
                    os.path.join(
                        paths["tgt_desc_output_dir"], f"target_{model_name}.txt"
                    ),
                )
            save_descriptions(
                adv_descriptions,
                os.path.join(
                    paths["desc_output_dir"], f"adversarial_{model_name}.txt"
                ),
            )
    
            print(f"Descriptions saved to {paths['desc_output_dir']}")
    
        except (FileNotFoundError, KeyError) as e:
            print(f"Error: {e}")
            return

        # finally:
        #     wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", nargs='+', required=True, type=str)
    parser.add_argument("--method", default="mattack", type=str)
    main()
