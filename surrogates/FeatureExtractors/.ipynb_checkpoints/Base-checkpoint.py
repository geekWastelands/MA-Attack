import torch
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Any, Callable, Dict
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel, AutoTokenizer

class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class EnsembleFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Tensor:
        # features = []
        # for model in self.extractors:
        #     features.append(model(x).squeeze())
        # features = torch.cat(features, dim=0)
        features = {}  # 不拼接，改为字典存储
        for i, model in enumerate(self.extractors):
            features[i] = model(x).squeeze()
        return features


class EnsembleFeatureLoss(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.previous_loss_list=[]

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor, y: Tensor=None):
    # def set_ground_truth(self, x: Tensor, y: List[str]=None, weight: float=0.3):
        # if len(self.ground_truth)>0:
        #     print(f'ground_truth:{self.ground_truth[0].shape}')
        self.ground_truth.clear()
        for model in self.extractors:
            if y is not None:
                self.ground_truth.append(model(x,y))
            else:
                self.ground_truth.append(model(x).to(x.device))

    def __call__(self, feature_dict: Dict[int, Tensor], weight: Any=[0.3, 0.5, 0.2], txt: bool = True) -> Tensor:
        loss = 0
        loss_idx = torch.zeros(size=(len(self.extractors),))
        loss_list = []
        for index, model in enumerate(self.extractors):
            if txt==False:
                gt = self.ground_truth[index]
                feature = feature_dict[index]
                loss += torch.mean(torch.sum(feature * gt, dim=1))
                # loss_list.append(torch.mean(torch.sum(feature * gt, dim=1)))
            else:
                gt_img, gt_txt = self.ground_truth[index]
                feature = feature_dict[index]
                
                fusion_txt = weight[index][1]*gt_txt[0] + weight[index][2]*gt_txt[1]
                fusion_txt = fusion_txt / fusion_txt.norm(dim=1, keepdim=True)
                
                fusion = fusion_txt + weight[index][0]*gt_img
                fusion = fusion / fusion.norm(dim=1, keepdim=True)
    
                # print(f"feature:{feature.shape} \t fusion:{fusion.shape}")
                cur_loss = torch.mean(torch.sum(feature * fusion, dim=1))
                loss_idx[index] = cur_loss
            # cur_loss = torch.mean(torch.sum(feature * gt_txt[0], dim=1))+torch.mean(torch.sum(feature * gt_txt[1], dim=1)) #+ 0.3*F.mse_loss(gt_img, adv_image)
                loss += cur_loss
        # print(f'loss:{loss}')
        # wt = F.softmax(-loss_idx/0.1, dim=0)
        # loss = torch.sum(wt * loss_idx)
        # 初始化 previous_loss_list（首次）
        # if len(self.previous_loss_list) == 0:
        #     self.previous_loss_list = [l.detach() for l in loss_list]

        # weights = []
        # for i in range(len(self.extractors)):
        #     ratio = loss_list[i].item() / (self.previous_loss_list[i].item() + 1e-8)
        #     weights.append(ratio)
        #     # 归一化 softmax 计算动态权重
        
        # T = 1.0
        # K = len(weights)
        # weights_np = np.array(weights)
        # weights_softmax = np.exp(weights_np / T)
        # weights_softmax /= np.sum(weights_softmax)
        # weights_softmax *= K  # 可选：缩放为 K


        # # 初始化 previous_loss_list（首次）
        # for i in range(len(self.extractors)):
        #     self.previous_loss_list[i] = loss_list[i].detach()

        # # 加权总损失
        # total_loss = sum(
        #     weights_softmax[i] * loss_list[i]
        #     for i in range(len(self.extractors))
        # )
        # return total_loss
        loss = loss / len(self.extractors)

        return loss

def select_k_hard_models(x: Tensor,y: List = None,tgt: Tensor = None,k: int = 3,eps: float = 16/255.0,beta: float = 0.8,device: str = 'cuda'):
    """
    静态贪心选出最难攻击的 k 个模型。
    难攻击 = 对 FGSM 扰动不敏感（vulnerability 小）。
    同时用梯度余弦相似度惩罚选出多样的模型。
    """
    from .ClipL336 import ClipL336FeatureExtractor
    from .ClipB16 import ClipB16FeatureExtractor
    from .ClipB32 import ClipB32FeatureExtractor
    from .ClipLaion import ClipLaionFeatureExtractor
    from .ClipL14 import ClipL14FeatureExtractor
    # from .ClipLaionB16 import ClipLaionB16FeatureExtractor
    from .ClipLaionB32 import ClipLaionB32FeatureExtractor
    from .ClipLaionH14 import ClipLaionH14FeatureExtractor
    from .ClipLaionL14 import ClipLaionL14FeatureExtractor
    
    extractors : Dict[str, type] = {
        "L336": ClipL336FeatureExtractor,
        "B16": ClipB16FeatureExtractor,
        "B32": ClipB32FeatureExtractor,
        "Laion": ClipLaionFeatureExtractor,
        "LaionH14": ClipLaionH14FeatureExtractor,
        "LaionB32": ClipLaionB32FeatureExtractor,
        "LaionL14": ClipLaionL14FeatureExtractor,
        # "LaionB16": ClipLaionB16FeatureExtractor,
        "L14": ClipL14FeatureExtractor
    }
    num_models = len(extractors)
    x = x.to(device)

    vulns, grads = [], []
    models = list(extractors.keys())
    for i, model_name in enumerate(models):
        model_class = extractors[model_name]
        model = model_class().eval().to(x.device).requires_grad_(False)

        (tgt_image_features, tgt_txt_features) = model(x=tgt, y=y)
        # print(model_name, ' ', tgt_txt_features)
        gt = tgt_txt_features[0]*0.4+tgt_txt_features[1]*0.3+tgt_image_features*0.3
        gt = gt / gt.norm(dim=1, keepdim=True)
        
        x_ = x.clone().detach().requires_grad_(True).to(x.device)
        feature = model(x_)
        loss = -torch.mean(torch.sum(feature * gt, dim=1))
        grad = torch.autograd.grad(loss, x_)[0]
        grads.append(F.normalize(grad.view(grad.size(0), -1).mean(0), dim=0))

        with torch.no_grad():
            x_adv = (x_ + eps * grad.sign()).clamp(0, 1)
            f_adv = model(x_adv)
            loss_after = -torch.mean(torch.sum(f_adv * gt, dim=1)).item()
            vulns.append(loss.item() - loss_after)

    selected, remaining = [], set(range(num_models))
    for _ in range(min(k, num_models)):
        best_idx, best_score = None, -1e9
        for i in remaining:
            diff = -vulns[i]  # 越小越难 -> 分数越高
            div_pen = 0.0
            if selected:
                sims = [F.cosine_similarity(grads[i].unsqueeze(0), grads[j].unsqueeze(0)).item()
                        for j in selected]
                div_pen = sum(sims) / len(sims)
            score = diff - beta * div_pen
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
        remaining.remove(best_idx)
        
    selected_idx = [list(extractors.keys())[i] for i in selected]
    print(f"[Select] Selected hardest {k} models: {selected}")
    return selected_idx

class CLSGuidedCompressor(nn.Module):
    def __init__(self, k: int = 16, threshold: float = 0.9):
        """
        k: top-k 模式下保留的 token 数
        threshold: energy-threshold 模式下的能量阈值 (0~1)
        """
        super().__init__()
        self.k = k
        self.threshold = threshold

    def forward(self, vision_outputs, hidden_states, mode: str = "topk"):
        """
        vision_outputs: CLIPVisionModel 的输出 (包含 attentions)
        hidden_states: [B, seq_len, 768]，最后一层隐状态
        mode: "topk" | "soft" | "energy"
        return: 压缩后的 token 表示 [B, N, 768] 或 [B, 768] (soft模式)
        """
        attn_last = vision_outputs.attentions[-1]   # [B, heads, seq_len, seq_len]
        cls_attn = attn_last[:, :, 0, :]            # [B, heads, seq_len]
        cls_attn_mean = cls_attn.mean(dim=1)        # [B, seq_len]

        patch_scores = cls_attn_mean[:, 1:]         # 去掉 CLS 本身 [B, seq_len-1]
        B, num_patches = patch_scores.shape

        if mode == "topk":
            # === Top-k 选择 ===
            topk_scores, topk_idx = torch.topk(patch_scores, self.k, dim=-1)  # [B, k]
            compressed_tokens = []
            for b in range(B):
                selected = hidden_states[b, topk_idx[b] + 1, :]  # +1 跳 CLS
                compressed_tokens.append(selected)
            compressed_tokens = torch.stack(compressed_tokens)  # [B, k, 768]
            return compressed_tokens

        elif mode == "soft":
            # === soft-weight pooling ===
            weights = patch_scores / patch_scores.sum(dim=-1, keepdim=True)  # [B, num_patches]
            pooled = torch.einsum("bp,bpd->bd", weights, hidden_states[:, 1:, :])  # [B,768]
            return pooled

        elif mode == "energy":
            # === Energy-threshold ===
            sorted_scores, sorted_idx = torch.sort(patch_scores, dim=-1, descending=True)  # [B,num_patches]
            cum_energy = torch.cumsum(sorted_scores, dim=-1) / torch.sum(sorted_scores, dim=-1, keepdim=True)  # [B,num_patches]

            compressed_tokens = []
            for b in range(B):
                # 找到能量超过 threshold 的位置
                valid = cum_energy[b] <= self.threshold
                num_keep = valid.sum().item()
                if num_keep < 1:  # 至少保留1个
                    num_keep = 1
                selected_idx = sorted_idx[b, :num_keep]
                selected = hidden_states[b, selected_idx + 1, :]
                compressed_tokens.append(selected)
            # 注意：不同 batch 可能选的数不一样 → 这里返回 list
            return compressed_tokens

        else:
            raise ValueError(f"Unknown mode {mode}, choose from ['topk','soft','energy']")
import numpy as np
from sklearn.decomposition import PCA
def fuse_pca(embeddings, n_components=1):
    """
    embeddings: list of [1,512] tensors
    return: [1,512] tensor
    """
    X = torch.cat(embeddings, dim=0).cpu().numpy()  # (n, 512)

    # PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)

    # 第一主成分方向
    fused = pca.components_[0]  # (512,)
    fused = fused / np.linalg.norm(fused)

    return torch.tensor(fused, dtype=torch.float32).unsqueeze(0).to(X.device)  # [1,512]

def fuse_svd(embeddings):
    """
    embeddings: list of [1,512] tensors
    return: [1,512] tensor
    """
    X = torch.cat(embeddings, dim=0).cpu().numpy()  # (n, 512)

    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    fused = Vt[0]   # (512,)
    fused = fused / np.linalg.norm(fused)

    return torch.tensor(fused, dtype=torch.float32).unsqueeze(0)  # [1,512]

def freq_fusion(txt_features, alpha=0.5, mode="fft"):
    """
    emb1, emb2: [1, 512] tensor
    alpha: 权重
    mode: "fft" / "dct"
    """
    if mode == "fft":
        f1 = torch.fft.fft(txt_features[0])
        f2 = torch.fft.fft(txt_features[1])
    elif mode == "dct":
        # 用 torch.fft.fft 实现 DCT-II 的近似
        f1 = torch.fft.rfft(txt_features[0])
        f2 = torch.fft.rfft(txt_features[1])
    else:
        raise ValueError("Unsupported mode")

    # 融合（这里简单加权）
    fused = alpha * f1 + (1 - alpha) * f2

    # 回到时域
    if mode == "fft":
        out = torch.fft.ifft(fused).real
    else:
        out = torch.fft.irfft(fused, n=emb1.shape[-1])

    # 归一化
    out = out / out.norm(dim=-1, keepdim=True)
    return out


import torch
from torch_dct import dct_2d, idct_2d

def lowpass_dct_filter(x: torch.Tensor, keep_ratio: float = 0.9) -> torch.Tensor:
    """
    对 [B, C, H, W] 图像进行 DCT 低通滤波，从频率中心开始保留低频。
    """
    assert x.ndim == 4, f"Expected [B, C, H, W], got {x.shape}"
    B, C, H, W = x.shape

    # 1️⃣ DCT 变换
    X_dct = dct_2d(x)  # [B, C, H, W]

    # 2️⃣ 构造低通掩码（左上角区域）
    mask = torch.zeros((1, 1, H, W), device=x.device, dtype=x.dtype)
    keep_h = int(H * keep_ratio)
    keep_w = int(W * keep_ratio)

    # 只保留左上角 [0:keep_h, 0:keep_w]
    mask[:, :, 0:keep_h, 0:keep_w] = 1.0
    mask = mask.expand(B, C, H, W)

    # 3️⃣ 应用掩码
    X_dct_filtered = X_dct * mask

    # 4️⃣ IDCT 逆变换
    x_filtered = idct_2d(X_dct_filtered)

    return x_filtered


def cfa_multi_text(X: torch.Tensor, Y1: torch.Tensor, Y2: torch.Tensor, k: int):
    """
    CFA 对齐一个图像模态 + 两个文本模态
    Args:
        X:  [n, d_x]
        Y1: [n, d_y1]
        Y2: [n, d_y2]
        k:  投影空间维度
    Returns:
        Zx:  [n, k] 图像模态投影结果
        Zy1: [n, k] 文本1模态投影结果
        Zy2: [n, k] 文本2模态投影结果
        A:   [d_x, k] 图像投影矩阵
        B1:  [d_y1, k] 文本1投影矩阵
        B2:  [d_y2, k] 文本2投影矩阵
    """
    assert X.shape[0] == Y1.shape[0] == Y2.shape[0], "All modalities must have the same number of samples"
    n = X.shape[0]

    # 中心化
    Xc = X - X.mean(dim=0, keepdim=True)
    Y1c = Y1 - Y1.mean(dim=0, keepdim=True)
    Y2c = Y2 - Y2.mean(dim=0, keepdim=True)

    # 拼接两个文本模态
    Y_all = torch.cat([Y1c, Y2c], dim=1)  # [n, d_y1 + d_y2]

    # 计算跨模态协方差矩阵
    cov = Xc.T @ Y_all / (n - 1)  # [d_x, d_y1+d_y2]

    # SVD
    U, S, Vh = torch.linalg.svd(cov, full_matrices=False)

    # 取前k维
    A = U[:, :k]              # [d_x, k]
    B_all = Vh.T[:, :k]       # [d_y1 + d_y2, k]

    # 拆回两个文本投影矩阵
    d_y1 = Y1.shape[1]
    d_y2 = Y2.shape[1]
    B1 = B_all[:d_y1, :]
    B2 = B_all[d_y1:, :]

    # 投影
    Zx = Xc @ A
    Zy1 = Y1c @ B1
    Zy2 = Y2c @ B2

    return Zx, Zy1, Zy2