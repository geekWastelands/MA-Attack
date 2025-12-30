import os
import json
import hashlib
import random
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision
from PIL import Image
import hydra
from omegaconf import DictConfig
import os
from config_schema import MainConfig
from functools import partial
from typing import List, Dict, Optional, Tuple
from torch import nn
# from pytorch_lightning import seed_everything
import wandb
import json
from omegaconf import OmegaConf
from tqdm import tqdm
# from surrogates.FeatureExtractors.Base import lowpass_dct_filter
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from surrogates import (
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    BlipcocoFeatureExtractor,
    BlipFeatureExtractor,
    ClipLaionMultiligualFeatureExtractor,
    ClipLaionH14FeatureExtractor,
    ClipLaionB32FeatureExtractor,
    ClipLaionL14FeatureExtractor,
    ClipLaionB16FeatureExtractor,
    ClipL14FeatureExtractor,
    EnsembleFeatureLoss,
    EnsembleFeatureExtractor,
    EnsembleFtExtractor
)

from utils import *
# (
#     hash_training_config, 
#     setup_wandb, 
#     ensure_dir, 
#     random_resized_center_crop, 
#     load_box_centers_from_json, 
#     _plot_and_save_bars,
#     _tensor_to_heatmap,
#     heatmap,
#     pack_feat
# )

def set_environment(seed=2023):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


BACKBONE_MAP: Dict[str, type] = {
    "L336": ClipL336FeatureExtractor,
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor,
    # "ViT": VisionTransformerFeatureExtractor,
    # "LaionM": ClipLaionMultiligualFeatureExtractor,
    # "BLIP": BlipFeatureExtractor,
    # "BLIPCOCO": BlipcocoFeatureExtractor
    "LaionH14": ClipLaionH14FeatureExtractor,
    "LaionB32": ClipLaionB32FeatureExtractor,
    "LaionL14": ClipLaionL14FeatureExtractor,
    "LaionB16": ClipLaionB16FeatureExtractor,
    "L14": ClipL14FeatureExtractor
}
grad_list = []
def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())

def apply_symlog(data):
    # 使用 symlog 变换放大微小差异
    return np.sign(data) * np.log1p(np.abs(data))


def compute_fluctuation_metrics(var_seq):
    # var_seq = np.array(var_seq)
    var_seq = np.array([float(v) for v in var_seq])
    # 1. First derivative magnitude
    d1 = np.abs(np.diff(var_seq))

    # 2. Second derivative magnitude
    d2 = np.abs(np.diff(var_seq, n=2))

    # 3. Moving std
    window = 20
    mov_std = np.array([np.std(var_seq[max(0, i-window):i+1]) 
                        for i in range(len(var_seq))])

    # 4. Fourier high-frequency energy
    F = np.fft.rfft(var_seq - var_seq.mean())
    power = np.abs(F)**2
    high_freq_energy = power[int(len(power)*0.3):].sum()  # top 70% frequencies

    # 5. Volatility index
    volatility = np.mean(d1)

    # 6. Total variation
    total_variation = d1.sum()
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(var_seq, marker='o')
    plt.title("Variance Sequence (Var_t)")
    plt.xlabel("Step")
    plt.ylabel("Variance")

    plt.subplot(2, 2, 2)
    plt.plot(d1, marker='o')
    plt.title("1st-order Difference |ΔVar|")
    plt.xlabel("Step")
    plt.ylabel("|Var_t - Var_(t-1)|")

    plt.subplot(2, 2, 3)
    plt.plot(d2, marker='o')
    plt.title("2nd-order Difference |Δ²Var|")
    plt.xlabel("Step")
    plt.ylabel("|ΔVar_t - ΔVar_(t-1)|")

    plt.subplot(2, 2, 4)
    plt.plot(mov_std, marker='o')
    plt.title("Moving Std of Var (Local Fluctuation)")
    plt.xlabel("Step")
    plt.ylabel("STD(Var)")

    plt.tight_layout()
    plt.show()
    # plt.savefig("./LAT/fig/var_seq_4.png", dpi=200); plt.close()
    print(f'high_freq_energy: {high_freq_energy}')
    return {
        "d1": d1,
        "d2": d2,
        "mov_std": mov_std,
        "high_freq_energy": high_freq_energy,
        "volatility": volatility,
        "total_variation": total_variation,
    }

def get_models(cfg: MainConfig, model_list: List[str]=None):
    """Get models based on configuration.

    Args:
        cfg: Configuration object containing model settings

    Returns:
        Tuple of (feature_extractor, list of models)

    Raises:
        ValueError: If ensemble=False but multiple backbones specified
    """
    if not cfg.model.ensemble and len(cfg.model.backbone) > 1:
        raise ValueError("When ensemble=False, only one backbone can be specified")
        
    # surrogate_model = ["L336", "B16", "B32", "Laion"]
    # cfg.model.backbone = random.sample(surrogate_model, 3)
    
    models = []
    if model_list is None:
        for backbone_name in cfg.model.backbone:
            if backbone_name not in BACKBONE_MAP:
                raise ValueError(
                    f"Unknown backbone: {backbone_name}. Valid options are: {list(BACKBONE_MAP.keys())}"
                )
            model_class = BACKBONE_MAP[backbone_name]
            model = model_class().eval().to(cfg.model.device).requires_grad_(False)
            models.append(model)
    else:
        for backbone_name in model_list:
            if backbone_name not in BACKBONE_MAP:
                raise ValueError(
                    f"Unknown backbone: {backbone_name}. Valid options are: {list(BACKBONE_MAP.keys())}"
                )
            model_class = BACKBONE_MAP[backbone_name]
            model = model_class().eval().to(cfg.model.device).requires_grad_(False)
            models.append(model)

    if cfg.model.ensemble:
        ensemble_extractor = EnsembleFtExtractor(models)
    else:
        ensemble_extractor = models[0]  # Use single model directly

    return ensemble_extractor, models


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)
def crop(img, txt):
    img = img[0]
    txt = txt[0]
    
def custom_data_loader(file_path, batch_size):
    if len(file_path)==3:
        with open(file_path[0], "r", encoding="utf-8") as f1, open(file_path[1], "r", encoding="utf-8") as f2, open(file_path[2], "r", encoding="utf-8") as f3:
            batch = []
            for line1, line2, line3 in zip(f1, f2, f3):
                path1 = line1.strip() 
                path2 = line2.strip() 
                path3 = line3.strip() 
                batch.append([path1, path2, path3])
                if len(batch) == batch_size:
                    yield batch  
                    batch = []  
            if batch:
                yield batch    
    elif len(file_path)==2:
        with open(file_path[0], "r", encoding="utf-8") as f1, open(file_path[1], "r", encoding="utf-8") as f2:
            batch = []
            for line1, line2 in zip(f1, f2):
                path1 = line1.strip() 
                path2 = line2.strip() 
                batch.append([path1, path2])
                if len(batch) == batch_size:
                    yield batch  
                    batch = []  
            if batch:
                yield batch   
    else:
        with open(file_path[0], 'r', encoding='utf-8') as file:
            batch = []
            for line in file:
                path = line.strip() 
                batch.append(path)
                
                if len(batch) == batch_size:
                    yield batch  
                    batch = []  
    
            if batch:
                yield batch    
            
def custom_json_loader(file_path, batch_size):
    with open(file_path[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
    data_sorted = sorted(data, key=lambda d: d["image"])
    batch = []
    for item in data_sorted:
        image_path = item["image"]
        caption = random.choice(item["caption"])
        batch.append(item["caption"])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
    
@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    set_environment()
    # run = setup_wandb(cfg)
    # ensemble_extractor = [
    #     ClipB16FeatureExtractor().eval().to(cfg.model.device).requires_grad_(False),
    #     ClipB32FeatureExtractor().eval().to(cfg.model.device).requires_grad_(False),
    #     ClipLaionFeatureExtractor().eval().to(cfg.model.device).requires_grad_(False),
    # ]
    # ensemble_loss = get_ensemble_loss(cfg, models)
    ensemble_extractor, models = get_models(cfg)

    transform_fn = transforms.Compose(
        [
            transforms.Resize(
                cfg.model.input_res,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(cfg.model.input_res),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Lambda(lambda img: to_tensor(img)),
        ]
    )

    clean_data = ImageFolderWithPaths(cfg.data.cle_data_path, transform=transform_fn)
    target_data = ImageFolderWithPaths(cfg.data.tgt_data_path, transform=transform_fn)

    data_loader_imagenet = torch.utils.data.DataLoader(
        clean_data, batch_size=cfg.data.batch_size, shuffle=False
    )
    data_loader_target = torch.utils.data.DataLoader(
        target_data, batch_size=cfg.data.batch_size, shuffle=False
    )

    if "txt" in cfg.data.tgt_txt_path[0]:
        tgt_text_loader = custom_data_loader(cfg.data.tgt_txt_path, 1)
    elif "json" in cfg.data.tgt_txt_path[0]:
        tgt_text_loader = custom_json_loader(cfg.data.tgt_txt_path, 1)
    else:
        target_txt_data = ImageFolderWithPaths(cfg.data.tgt_txt_path, transform=transform_fn)
    
        tgt_text_loader = torch.utils.data.DataLoader(
            target_txt_data, batch_size=cfg.data.batch_size, shuffle=False
        )
    cle_text_loader = custom_data_loader(cfg.data.cle_txt_path, 1)

    source_crop = (
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
        if cfg.model.use_source_crop
        else torch.nn.Identity()
    )
    target_crop = (
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
        if cfg.model.use_target_crop
        else torch.nn.Identity()
    )
    json_path = "/root/autodl-tmp/codes/M-Attack/resources/test.json"
    tgt_centers_list = load_box_centers_from_json(json_path)
    
    # json_path = "/root/autodl-tmp/codes/M-Attack/resources/bbox_source.json"
    # org_centers_list = load_box_centers_from_json(json_path)
    # print(centers_list)
    
    data_iter = zip(data_loader_imagenet, data_loader_target, tgt_text_loader, cle_text_loader)

     # cfg.config_hash
    num_dirs = 1
    config_hash = cfg.config_hash # f"ens_multi_crop_{num}"
    print(f'hash:{config_hash}')
    for i, ((image_org, _, path_org), (image_tgt, _, path_tgt), tgt_text, cle_text) in enumerate(
        tqdm(list(data_iter), total=len(data_loader_imagenet))
    ):
        # print(f"path_org:{path_org}  path_tgt:{path_tgt}  tgt_text:{tgt_text} tgt_centers_list:{len(tgt_centers_list[i])} org_centers_list:{len(org_centers_list[i])}")
        if cfg.data.batch_size * (i + 1) > cfg.data.num_samples:
            break
                
        print(f"\nProcessing image {i+1}/{cfg.data.num_samples//cfg.data.batch_size}")
                
        attack_imgpair(
            cfg=cfg,
            ensemble_extractor=ensemble_extractor,
            source_crop=source_crop,
            img_index=i,
            image_org=image_org,
            path_org=path_org,
            image_tgt=image_tgt,
            target_crop=target_crop,
            cle_text=cle_text,
            tgt_text=tgt_text,
            config_hash=config_hash,
            target_centers_list=tgt_centers_list,
            num_dirs=num_dirs,
        )
    # np.save('mattack_var_1.npy', grad_list)
    # wandb.finish()

def attack_imgpair(
    cfg: MainConfig,
    ensemble_extractor: List[nn.Module],
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    path_org: List[str],
    image_tgt: torch.Tensor,
    cle_text: List[str],
    tgt_text: List[str],
    config_hash: str,
    target_centers_list: Optional[List[List[Tuple[int,int]]]]=None,
    num_dirs: int=1,

):
    image_org, image_tgt = image_org.to(cfg.model.device), image_tgt.to(
        cfg.model.device
    )
    adv_image = multi_direction_grad(
        cfg = cfg,
        model_extractor = ensemble_extractor, 
        img_index = img_index,
        image_tgt       = image_tgt,         # (1,C,H,W)
        image_org        = image_org, 
        target_crop     = target_crop,
        num_dirs        = num_dirs,
        target_centers_list=target_centers_list,
        tgt_text = tgt_text,
    )
    # Save images
    for path_idx in range(len(path_org)):
        folder, name = (
            path_org[path_idx].split("/")[-2],
            path_org[path_idx].split("/")[-1],
        )
        # Use config hash in output path
        folder_to_save = os.path.join(cfg.data.output, "img", config_hash, folder)
        ensure_dir(folder_to_save)

        if "JPEG" in name:
            torchvision.utils.save_image(
                adv_image[path_idx], os.path.join(folder_to_save, name[:-4]) + "png"
            )
        elif "png" in name:
            torchvision.utils.save_image(
                adv_image[path_idx], os.path.join(folder_to_save, name)
            )
    '''
    # 1. 运行多次实验并收集数据
    all_trajectories = []  # 存储各组的 (deltas, losses, label)
    dir_settings = [1, 4] # 对比不同的 num_dirs
    colors = ['green', 'blue']
    labels = ['M-Attack', 'Ours']
    for n_dir in dir_settings:    
        adv_image, adv_history, loss_history = multi_direction_grad(
            cfg = cfg,
            model_extractor = ensemble_extractor, 
            img_index = img_index,
            image_tgt       = image_tgt,         # (1,C,H,W)
            image_org        = image_org, 
            target_crop     = target_crop,
            num_dirs        = n_dir,
            target_centers_list=target_centers_list,
            tgt_text = tgt_text
        )
        all_trajectories.append((adv_history, loss_history))
    all_deltas_concat = np.concatenate([t[0] for t in all_trajectories], axis=0)
    all_deltas_concat = apply_symlog(all_deltas_concat)
    scaler = StandardScaler()
    all_deltas_scaled = scaler.fit_transform(all_deltas_concat)
    pca = PCA(n_components=2)
    pca.fit(all_deltas_scaled)

    # --- 3. 开始绘图：创建两个子图 ---
    # 创建一个 1x2 的布局
    fig = plt.figure(figsize=(16, 6))
    # 左边：3D 子图
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # 右边：2D 子图
    ax2 = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.45, right=0.55, wspace=0.15)
    for i, (d_hist, l_hist) in enumerate(all_trajectories):
        # --- A. 处理 2D 轨迹数据 ---
        d_log = apply_symlog(d_hist)
        coords_2d = pca.transform(d_log)
        
        # 截取前 250 步并对齐起点
        coords_2d = coords_2d[:250, :]
        start_coord = coords_2d[0]
        x = coords_2d[:, 0] - start_coord[0]
        y = coords_2d[:, 1] - start_coord[1]
        z = l_hist[:250]
        # 绘制轨迹
        ax1.plot(x, y, z, color=colors[i], label=labels[i], marker='o', markersize=0.5, alpha=0.7)
        # 标记终点
        ax1.scatter(x[-1], y[-1], z[-1], color='red', marker='x', s=30, zorder=5)
        
        # --- B. 绘制左图：2D PCA 轨迹图 ---
        # ax1.plot(x, y, color=colors[i], label=labels[i], marker='o', markersize=2, alpha=0.6, linewidth=1)
        # ax1.scatter(x[-1], y[-1], color='black', s=30, zorder=5) # 终点
        ax1.scatter(0, 0, 0, color='black', marker='o', s=40, zorder=5) # 共同起点
        
        # --- C. 绘制右图：Loss 随 Step 变化图 ---
        steps = np.arange(len(l_hist))
        ax2.plot(steps, l_hist, color=colors[i], label=labels[i], linewidth=2)

    # --- 4. 修饰图表 ---

    # 修饰左图 (PCA)
    ax1.set_xlabel('PCA Component 1', fontsize=12)
    ax1.set_ylabel('PCA Component 2', fontsize=12)
    ax1.grid(False)

    # 隐藏 pane
    ax1.zaxis.pane.set_visible(False)
    ax1.xaxis.pane.set_alpha(0)
    ax1.yaxis.pane.set_alpha(0)

    # 去掉 z 轴元素
    ax1.set_zlabel('')
    ax1.set_zticks([])
    ax1.zaxis.line.set_alpha(0)
    # ax1.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)   # alpha = 0
    # ax1.zaxis._axinfo["grid"]["linewidth"] = 0
    # ax1.zaxis.line.set_alpha(0)
    
    ax1.set_title('Attack Trajectories in Image Space (PCA)', fontsize=14)
    # ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    ax1.view_init(elev=90, azim=-90)

    # 修饰右图 (Loss)
    ax2.set_xlabel('Iteration Steps', fontsize=12)
    ax2.set_ylabel('Loss (Similarity)', fontsize=12)
    ax2.set_title('Loss Convergence Comparison', fontsize=14)
    # ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()

    # 保存结果
    save_path = f'./LAT/fig/dual_plot_{img_index}.png'
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    '''

def multi_direction_grad(
    cfg: MainConfig,
    model_extractor,     # <---- model_extractor: 一个整体（内部包含若干子模型）
    img_index: int,
    image_org, 
    image_tgt, 
    target_crop,
    num_dirs=1,
    target_centers_list=None,   # 多模型加权
    tgt_text=None
):

    device = image_org.device
    B = num_dirs
    loss_vec = []
    grad_var = []
    pbar = tqdm(range(cfg.optim.steps), desc=f"Attack progress")

    centers_list = target_centers_list[img_index] # + source_centers_list[img_index]
    # print(f'centers_list:{centers_list}')
    use_center = (random.random() < 0.1)

    if len(centers_list)==0:
        source_centers=(0.5,0.5)
        tgt_centers = (0.5,0.5)
    else:
        source_centers = random.choice(centers_list)
        tgt_centers = random.choice(centers_list)
        # print(f'tgt center:{tgt_centers} source_centers:{source_centers}')

    # trajectory_adv = [] 
    # trajectory_loss = []

    delta = torch.zeros_like(image_org, requires_grad=True)
    for epoch in pbar:
        # 记录日志
        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        tgt_list = []
        if random.random() < 0.:
            for _ in range(num_dirs):
                tgt_crop = random_resized_center_crop(
                    image_tgt, centers=tgt_centers,
                    output_size=cfg.model.input_res,
                )
                tgt_list.append(tgt_crop)
        else:
            for _ in range(num_dirs):
                tgt_list.append(target_crop(image_tgt))   # 每次都不同 crop
        tgt_cat = torch.cat(tgt_list, dim=0)
        tgt_feat = model_extractor(tgt_cat)
        # if cfg.data.use_tgt_txt:
        #     tgt_txts = tgt_text*num_dirs
        #     tgt_feat = model_extractor(tgt_cat, tgt_txts) # (3,4,(img,txt))
        # else:
        #     tgt_feat = model_extractor(tgt_cat) # (len(extractors), num_dirs, img_feat)
            # print(f'tgt:{tgt_feat[0].shape}')
        # print(f'tgt:{len(tgt_list)}')
        # tgt_feat = torch.stack(tgt_list, dim=0)
        adv = image_org + delta

        adv_list = []
        if random.random() < 0.:
            for _ in range(num_dirs):
                adv_crop = random_resized_center_crop(
                    adv, centers=tgt_centers,
                    output_size=cfg.model.input_res,
                )
                adv_list.append(adv_crop)
        else:    
            for _ in range(num_dirs):
                adv_list.append(target_crop(adv))   # 每次都不同 crop
    
        adv_cat = torch.cat(adv_list, dim=0)    # [M,3,H,W]
        # feats = torch.stack(adv_list, dim=0)
        feats = model_extractor(adv_cat)        # [M,D]
        loss = 0.
        for ft, tgt_ft in zip(feats, tgt_feat):
            loss += torch.mean(torch.sum(ft * tgt_ft))
        loss = loss/(num_dirs*3)
        metrics["similarity"] = loss.item()/num_dirs

        cur_adv = delta
        # cur_adv = torch.clamp(cur_adv / 255.0, 0, 1)
        # trajectory_adv.append(cur_adv.detach().cpu().numpy().flatten().tolist()) 
        # trajectory_loss.append(loss.item())
        # print(f'feat:{feats}')

        # adv_feats = pack_feat(adv_list)   # [B, 2048]
        # tgt_feats = pack_feat(tgt_list)   # [B, 2048]
        # loss = torch.sum(adv_feats @ tgt_feats.T)
        # loss = torch.einsum('bd,cd->', adv_feats, tgt_feats)
        # loss_list = []
        # for ft in adv_list:
        #     # print(f'ft:{ft}')
        #     for tgt_ft in tgt_list:
        #         # print(f'ft:{ft[0].shape}  len:{len(ft)}')
        #         loss = 0.
        #         for f, tf in zip(ft, tgt_ft):
        #             loss+=torch.mean(torch.sum(f * tf))
        #         loss_list.append(loss)
        # loss_list = sum(loss_list)
        # loss_list = sum(loss_list)/len(loss_list)
        # sorted_list = sorted(loss_list, reverse=True)
        # loss_list = sum(sorted_list[-4:])
        # metrics["similarity"] = loss.item()/num_dirs
        # loss_vec.append(metrics["similarity"])
        grad = torch.autograd.grad(
            outputs=loss,
            inputs=delta,
            create_graph=False
        )[0]
        
        # log_metrics(pbar, metrics, img_index, epoch)

        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(grad),
            -cfg.optim.epsilon,
            cfg.optim.epsilon
        )
        
    # log_loss(trajectory_loss, img_index, run)
    # print(f'var_grad:{grad_var}')
    # grad_list.append(np.array(grad_var))
    # compute_fluctuation_metrics(grad_var)
    adv = image_org + delta
    adv = torch.clamp(adv / 255.0, 0, 1)
    # np.save(f'./LAT/npy/mattack_{img_index}.npy', np.array(trajectory_adv))
    # data_json = {
    #     "delta": trajectory_adv,
    #     "loss": trajectory_loss
    # }
    # filename_json = f'./LAT/npy/multi_crop_attack_{img_index}.json'
    # with open(filename_json, 'w', encoding='utf-8') as f:
    #     json.dump(data_json, f, ensure_ascii=False, indent=4) # indent=4增加可读性
    return adv

def log_loss(trajectory_loss, img_index, run):
    table = wandb.Table(columns=["attack_step", "loss"])

    for t, loss in enumerate(trajectory_loss):
        table.add_data(t, float(loss))

    run.log({
        f"loss_curve/img_{img_index}": table
    })

def log_metrics(pbar, metrics, img_index, epoch=None):
    """
    Log metrics to progress bar and wandb.

    Args:
        pbar: tqdm progress bar to update
        metrics: Dictionary of metrics to log
        img_index: Index of the image (for wandb logging)
        epoch: Optional epoch number for logging
    """
    # Format metrics for progress bar
    pbar_metrics = {
        k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" for k, v in metrics.items()
    }
    pbar.set_postfix(pbar_metrics)

if __name__ == "__main__":
    main()