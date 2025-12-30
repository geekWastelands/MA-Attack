import torch
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel, AutoTokenizer
from .Base import * # BaseFeatureExtractor, CLSGuidedCompressor, fuse_pca, fuse_svd, freq_fusion, lowpass_dct_filter
from torchvision import transforms
import numpy as np
from sklearn.decomposition import PCA
import torch.nn.functional as F

class ClipLaionB32FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(ClipLaionB32FeatureExtractor, self).__init__()
        self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
        self.tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
        self.normalizer = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )

    def forward(self, x, y=None, wt=0.3):
        # x = lowpass_dct_filter(x)
        inputs = dict(pixel_values=self.normalizer(x))
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        if y is not None:
            if isinstance(y[0], list):
                y = [str(item) for sub in y for item in sub]
            else:
                y = [str(item) for item in y]
            # txt_features = torch.zeros(1, 512).to(x.device)
            txt_features = []
            for y_i in y:
                inputs = self.tokenizer(y_i, 
                    padding=True, 
                    truncation=True, 
                    max_length=77,           # CLIP 文本 token 最大长度
                    return_tensors="pt").to(x.device)
                    # print(inputs["input_ids"].size())
                txt_feature = self.model.get_text_features(input_ids=inputs["input_ids"])
                txt_feature = txt_feature / txt_feature.norm(dim=1, keepdim=True)
                txt_features.append(txt_feature)
            txt_features = torch.stack(txt_features, dim=0)
            return (image_features.to(x.device), txt_features.to(x.device))
            # return fusion_features
        return image_features