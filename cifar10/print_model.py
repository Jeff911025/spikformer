import torch
from timm.models import create_model
import model  
# 用你自己在 train.py 里用的那些超参
model = create_model(
    'spikformer',
    pretrained=False,
    embed_dims=384,
    num_heads=12,
    mlp_ratios=4,
    depths=4,
    sr_ratios=1,
    img_size_h=32, img_size_w=32,
    patch_size=4,
    in_channels=3,
    num_classes=10,
    T=4,
)

# 直接打印整个模型结构
print(model)
