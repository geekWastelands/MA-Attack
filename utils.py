"""Shared utilities for adversarial attack and text generation models."""

import os
import json
import yaml
import hashlib
import base64
from typing import Dict, Any, List, Union
from omegaconf import OmegaConf
import wandb
from config_schema import MainConfig

import re
from pathlib import Path
from openai import OpenAI
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_api_keys() -> Dict[str, str]:
    """Load API keys from the api_keys file.
    
    Returns:
        Dict[str, str]: Dictionary containing API keys for different models
        
    Raises:
        FileNotFoundError: If no api_keys file is found
    """
    # yaml.safe_load(".")
    for ext in ['yaml', 'yml', 'json']:
        file_path = f'./api_keys.{ext}'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                if ext in ['yaml', 'yml']:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
    
    raise FileNotFoundError(
        "API keys file not found. Please create api_keys.yaml, api_keys.yml, or api_keys.json "
        "in the root directory with your API keys."
    )


def get_api_key(model_name: str) -> str:
    """Get API key for specified model.
    
    Args:
        model_name: Name of the model to get API key for
        
    Returns:
        str: API key for the specified model
        
    Raises:
        KeyError: If API key for model is not found
    """
    api_keys = load_api_keys()
    if model_name not in api_keys:
        raise KeyError(
            f"API key for {model_name} not found in api_keys file. "
            f"Available models: {list(api_keys.keys())}"
        )
    return api_keys[model_name]


def hash_training_config(cfg: MainConfig) -> str:
    """Create a deterministic hash of training-relevant config parameters.
    
    Args:
        cfg: Configuration object containing model settings
        
    Returns:
        str: MD5 hash of the config parameters
    """
    # Convert backbone list to plain Python list
    if isinstance(cfg.model.backbone, (list, tuple)):
        backbone = list(cfg.model.backbone)
    else:
        backbone = OmegaConf.to_container(cfg.model.backbone)
        
    # Create config dict with converted values
    train_config = {
        "data": {
            "batch_size": int(cfg.data.batch_size),
            "num_samples": int(cfg.data.num_samples),
            "cle_data_path": str(cfg.data.cle_data_path),
            "tgt_data_path": str(cfg.data.tgt_data_path),
            "use_tgt_txt":str(cfg.data.use_tgt_txt)
        },
        "optim": {
            "alpha": float(cfg.optim.alpha),
            "epsilon": int(cfg.optim.epsilon),
            "steps": int(cfg.optim.steps),
        },
        "model": {
            "input_res": int(cfg.model.input_res),
            "use_source_crop": bool(cfg.model.use_source_crop),
            "use_target_crop": bool(cfg.model.use_target_crop),
            "crop_scale": tuple(float(x) for x in cfg.model.crop_scale),
            "ensemble": bool(cfg.model.ensemble),
            "backbone": backbone,
        },
        "attack": cfg.attack,
    }
    
    # Convert to JSON string with sorted keys
    json_str = json.dumps(train_config, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()


def setup_wandb(cfg: MainConfig, tags=None) -> None:
    """Initialize Weights & Biases logging.
    
    Args:
        cfg: Configuration object containing wandb settings
    """
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    return wandb.init(
        project=cfg.wandb.project,
        config=config_dict,
        tags=tags,
    )


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def apply_symlog(data):
    # 使用 symlog 变换放大微小差异
    return np.sign(data) * np.log1p(np.abs(data))
    
def encode_image_to_base64(filepath):
    """
    将图片编码为 Base64 数据 URI 格式
    
    该函数读取本地图片文件，将其转换为 Base64 编码的数据 URI，
    这是一种可以直接嵌入到 JSON 或 HTML 中的图片表示格式。
    
    参数说明：
        filepath (str): 图片文件的本地路径
        
    返回值：
        str: 格式为 "data:image/jpeg;base64,..." 的完整数据 URI 字符串
        
    示例：
        >>> encode_image_to_base64("example.jpg")
        'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA...'
    """
    # 获取文件扩展名并转换为小写（如 ".jpg", ".png"）
    ext = os.path.splitext(filepath)[1].lower()
    
    # 定义文件扩展名到 MIME 类型的映射表
    # MIME 类型用于标识数据的格式，浏览器和 API 依赖此信息正确处理图片
    mime_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    
    # 根据扩展名获取对应的 MIME 类型，未知格式使用通用二进制类型
    mime_type = mime_map.get(ext, 'application/octet-stream')
    
    # 以二进制模式读取图片文件
    with open(filepath, "rb") as f:
        # 执行编码流程：
        # 1. f.read() - 读取文件的所有二进制数据
        # 2. base64.b64encode() - 将二进制数据编码为 Base64
        # 3. .decode('utf-8') - 将 Base64 字节串转换为 UTF-8 字符串
        # 4. f"data:{mime_type};base64,..." - 构造符合标准的数据 URI
        return f"data:{mime_type};base64,{base64.b64encode(f.read()).decode('utf-8')}"

        
def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure exists
    """
    os.makedirs(path, exist_ok=True)


def get_output_paths(cfg: MainConfig, config_hash: str) -> Dict[str, str]:
    """Get dictionary of output paths based on config.
    
    Args:
        cfg: Configuration object
        config_hash: Hash of training config
        
    Returns:
        Dict[str, str]: Dictionary containing output paths
    """
    return {
        'output_dir': os.path.join(cfg.data.output, "img", config_hash),
        'desc_output_dir': os.path.join(cfg.data.output, "description", config_hash)
    } 


import random
import torch
import torchvision.transforms.functional as TF
from typing import Optional, List, Tuple
from tqdm import tqdm
import torch.nn as nn

import math


def random_resized_center_crop(
    image: torch.Tensor,                     # [C,H,W]
    centers: Optional[Tuple[float, float]],  # (cx, cy) in relative coords [0,1]
    scale_range: Tuple[float, float] = (0.5, 0.9),
    ratio_range: Tuple[float, float] = (0.75, 1.33),
    output_size: int = 224,
) -> torch.Tensor:

    _, C, H, W = image.shape
    area = H * W

    # ----------------------
    # 1) 闅忔満閫変竴涓腑蹇冪偣
    # ----------------------
    cx_rel, cy_rel = centers # random.choice(centers)
    cx = int(cx_rel * W)
    cy = int(cy_rel * H)
    # if centers and len(centers) > 0:
    #     # 闅忔満閫変竴涓浉瀵瑰潗鏍?    #     cx_rel, cy_rel = random.choice(centers)
    #     cx = int(cx_rel * W)
    #     cy = int(cy_rel * H)
    # else:
    #     cx, cy = W // 2, H // 2

    scale_min, scale_max = scale_range
    ratio_min, ratio_max = ratio_range

    # ----------------------
    # 2) 澶氭灏濊瘯鎵惧埌鍚堟硶鐨勮鍓
    # ----------------------
    for attempt in range(10):
        target_area = random.uniform(scale_min, scale_max) * area
        log_ratio = (math.log(ratio_min), math.log(ratio_max))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        # 蹇呴』灏忎簬鍘熷浘
        if w <= W and h <= H:
            # ----------------------
            # 浠?center = (cx, cy) 涓哄浐瀹氫腑蹇?            # ----------------------
            x1 = cx - w // 2
            y1 = cy - h // 2
            x2 = x1 + w
            y2 = y1 + h

            # ----------------------
            # 鑷姩淇濊瘉涓嶈秺鐣岋紙骞崇Щ绐楀彛锛?            # ----------------------
            if x1 < 0:
                x1 = 0
                x2 = w
            if y1 < 0:
                y1 = 0
                y2 = h
            if x2 > W:
                x2 = W
                x1 = W - w
            if y2 > H:
                y2 = H
                y1 = H - h

            # 鍐嶆 clamp 闃叉寮傚父
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(x1 + 1, min(x2, W))
            y2 = max(y1 + 1, min(y2, H))

            # ----------------------
            # 鎵ц瑁佸壀
            # ----------------------
            crop = TF.crop(image, top=y1, left=x1, height=y2 - y1, width=x2 - x1)
            resized = TF.resize(crop, [output_size, output_size],
                                interpolation=TF.InterpolationMode.BICUBIC)
            return resized

    # ======================================================
    # Fallback锛?0娆″け璐ワ級锛屼娇鐢ㄤ腑蹇冭鍓紝淇濇寔鍜?torchvision 涓€鑷?    # ======================================================
    in_ratio = W / H
    if in_ratio < ratio_min:
        w = W
        h = int(round(w / ratio_min))
    elif in_ratio > ratio_max:
        h = H
        w = int(round(h * ratio_max))
    else:
        w, h = W, H

    x1 = (W - w) // 2
    y1 = (H - h) // 2

    crop = TF.crop(image, y1, x1, h, w)
    resized = TF.resize(crop, [output_size, output_size],
                        interpolation=TF.InterpolationMode.BICUBIC)
    return resized


def load_box_centers_from_json(json_path: str) -> List[Tuple[str, List[Tuple[int, int]]]]:
    """
    璇诲彇瀛樺偍姣忓紶鍥剧墖box涓績鐐瑰潗鏍囩殑JSON鏂囦欢銆?
    Args:
        json_path (str): JSON鏂囦欢璺緞銆?
    Returns:
        List[Tuple[str, List[Tuple[int, int]]]]:
            姣忎釜鍏冪礌鏄?(image_name, centers)锛?            鍏朵腑 centers 鏄?[(x, y), ...] 鏍煎紡鐨勫潗鏍囧垪琛ㄣ€?    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for image_name, centers in data.items():
        if not isinstance(centers, list):
            raise ValueError(f"Invalid format for {image_name}: expected list, got {type(centers)}")
        
        # image_centers = []
        # for c in centers:
        #     image_centers.append((float(c[0]), float(c[1])))
            # if isinstance(c, (list, tuple)) and len(c) == 2:
                # image_centers.append((float(c[0]), float(c[1])))
            # else:
            #     raise ValueError(f"Invalid coordinate format in {image_name}: {c}")
        
        results.append(centers)
    
    return results

import matplotlib.pyplot as plt
import numpy as np
def _plot_and_save_bars(vec, xlabel, ylabel, path):
    plt.figure(figsize=(5,2.5))
    plt.bar(np.arange(50), np.array(vec[:50]))
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _tensor_to_heatmap(t):
    # t: (1,C,H,W) or (C,H,W)
    tt = t.detach().cpu()
    if tt.dim() == 4:
        tt = tt[0]
    img = torch.norm(tt, dim=0)
    img = img - img.min()
    img = img / (img.max() + 1e-12)
    return img.numpy()

def save_multi_step_heatmaps(deltas, save_path="multi_steps.png"):
    """
    deltas: List[Tensor], e.g. [delta_step1, delta_step11, ...]
            每个 tensor shape (1,3,H,W)
    """
    n = len(deltas)
    plt.figure(figsize=(3*n, 3))

    for i, d in enumerate(deltas):
        if d.ndim == 4:
            d = d[0]
        d = d.norm(dim=0)
        d = d.detach().cpu().numpy()

        plt.subplot(1, n, i+1)
        plt.imshow(d, cmap="viridis")
        plt.title(f"Step {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def heatmap(delta):
    """Convert delta to norm heatmap"""
    d = delta.detach().cpu().numpy()[0]
    d = np.linalg.norm(d, axis=0)   # H×W
    d = d / (d.max() + 1e-8)
    return d

import numpy as np
import matplotlib.pyplot as plt

# x1, x2: shape = [100, 300]
# 例如：x1 = np.array(list1); x2 = np.array(list2)

def plot_envelope(x, label, color):
    # x.shape = [N_samples, T_steps]
    x = np.array(x)
    T = x.shape[1]
    iters = np.arange(T)

    x_min = x.min(axis=0)
    x_max = x.max(axis=0)

    plt.fill_between(iters, x_min, x_max, alpha=0.2, color=color)
    plt.plot(iters, x.mean(axis=0), color=color, label=label)

def pack_feat(feat_list):
    # feat_list: [ [f1,f2,f3], ... ]
    return torch.stack([
        torch.cat([f.flatten() for f in ft], dim=-1)
        for ft in feat_list
    ], dim=0)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def plot_attack_3d(adv_history, loss_history, img_index, label="Attack Path", color="blue"):
    """
    adv_history: [Steps, Dimensions] 的 numpy 数组
    loss_history: [Steps] 的 numpy 数组
    """
    # 1. PCA 降维：将高维图像向量降至 2 维
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(adv_history) # 得到 [Steps, 2]
    
    # 2. 准备 3D 数据
    x = coords_2d[:, 0]
    y = coords_2d[:, 1]
    z = loss_history

    # 3. 绘图
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹线
    ax.plot(x, y, z, color=color, marker='o', markersize=4, label=label, alpha=0.8)
    
    # 起点和终点标注
    ax.scatter(x[0], y[0], z[0], color='green', s=50, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=50, label='End')

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('Loss')
    ax.set_title('Adversarial Attack Trajectory in 3D')
    ax.legend()
    plt.savefig(f'./LAT/fig/3D_mattack_{img_index}.png', dpi=300, bbox_inches="tight")
    plt.show()


import torch.nn.functional as F
def calculate_similarity_distribution(feat1, feat2):
    """
    计算 feat1 和 feat2 中对应样本的余弦相似度分布 (Sim(feat1[i], feat2[i]))
    
    Args:
        feat1 (torch.Tensor): 尺寸 [B, D]
        feat2 (torch.Tensor): 尺寸 [B, D]
        
    Returns:
        np.array: 尺寸 [B,] 的相似度值数组
    """
    # dim=1 表示在特征维度 D 上计算相似度
    # 结果是一个尺寸为 [B] 的向量
    similarity_tensor = F.cosine_similarity(feat1, feat2, dim=1)
    return similarity_tensor.cpu().numpy()


if __name__ == "__main__":
    x1 = np.load('mattack_var_1.npy').tolist()
    x2 = np.load('mattack_var_4.npy').tolist()
    x1_150 = [row[100:] for row in x1]   # 形状变为 [100, 150]
    x2_150 = [row[100:] for row in x2]
    plt.figure(figsize=(10,5))
    plot_envelope(x1_150, "M-Attack", "tab:blue")
    plot_envelope(x2_150, "Ours", "tab:orange")

    plt.xlabel("Iteration")
    plt.ylabel("Variance")
    plt.title("Coverage Range of Variance Across Samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./LAT/fig/var.png") 
    plt.show()
