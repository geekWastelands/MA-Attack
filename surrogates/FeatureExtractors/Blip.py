import torch
from transformers import (
    Blip2VisionModel,
    Blip2VisionConfig,
    Blip2Processor,
    Blip2Model,
    BlipImageProcessor,
)
from torchvision import transforms
from .Base import BaseFeatureExtractor
device = "cuda" if torch.cuda.is_available() else "cpu"
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


class BlipFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(BlipFeatureExtractor, self).__init__()
        self.normalizer = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )
        self.processor = BlipImageProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.device = torch.device("cuda")
        self.eval().requires_grad_(False)

    def forward(self, x, y=None, wt=0.3):
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"].to(device)
        outputs = self.model.get_image_features(**inputs)
        pooler_output = outputs.pooler_output
        image_features = pooler_output / pooler_output.norm(dim=1, keepdim=True)
        
        if y is not None:
            if isinstance(y[0], list):
                y = [str(item) for sub in y for item in sub]
            else:
                y = [str(item) for item in y]
            # txt_features = torch.zeros(1, 1024).to(x.device)
            txt_features = []
            if len(y)>1:
                for y_i in y:
                    inputs = self.processor(text=y_i, return_tensors="pt").to(x.device)
                    # inputs = self.tokenizer(y_i, 
                    #     padding=True, 
                    #     truncation=True, 
                    #     max_length=77,           # CLIP 文本 token 最大长度
                    #     return_tensors="pt").to(x.device)
                    # print(inputs["input_ids"].size())
                    # txt_feature = self.model.get_text_features(input_ids=inputs["input_ids"])
                    # txt_feature = txt_feature / txt_feature.norm(dim=1, keepdim=True)
                    # txt_features.append(txt_feature)

                    outputs = self.model.language_model(**inputs, output_hidden_states=True)
                    txt_feature = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 特征
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
