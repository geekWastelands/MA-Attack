import torch
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel, AutoTokenizer
from .Base import BaseFeatureExtractor
from torchvision import transforms


class ClipLaionMultiligualFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(ClipLaionMultiligualFeatureExtractor, self).__init__()
        self.model = CLIPModel.from_pretrained("sentence-transformers/clip-ViT-B-32-multilingual-v1")
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/clip-ViT-B-32-multilingual-v1")
        self.normalizer = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # CLIP imgs mean and std.
        ]
    )

    def forward(self, x, y=None, wt=0.3):
        # x = torch.clamp(x, min=0, max=1)
        inputs = dict(pixel_values=self.normalizer(x))
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        if y is not None:
            if isinstance(y[0], list):
                y = [str(item) for sub in y for item in sub]
            else:
                y = [str(item) for item in y]
            # txt_features = torch.zeros(1, 1024).to(x.device)
            txt_features = []
            if len(y)>1:
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
                # txt_features = fuse_svd(txt_features).to(x.device)
                    # txt_features = txt_features + (1-wt)/len(y) * txt_feature
            else:
                # x = lowpass_dct_filter(x)
                inputs = self.tokenizer(y, 
                        padding=True, 
                        truncation=True, 
                        max_length=77,           # CLIP 文本 token 最大长度
                        return_tensors="pt").to(x.device)
                txt_feature = self.model.get_text_features(input_ids=inputs["input_ids"])
                txt_feature = txt_feature / txt_feature.norm(dim=1, keepdim=True)
                txt_features = txt_features + (1-wt) * txt_feature
                       
            txt_features = 0.4*txt_features[0]+0.3*txt_features[1]
            
            txt_features = txt_features / txt_features.norm(dim=1, keepdim=True)
            fusion_features = wt * image_features + txt_features
            # fusion_features = wt * image_features + (1-wt) * txt_features
            fusion_features = fusion_features / fusion_features.norm(dim=1, keepdim=True)
            # return txt_features
            # return (image_features.to(x.device), txt_features.to(x.device))
            return fusion_features
        return image_features
