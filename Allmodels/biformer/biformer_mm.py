
import torch
import torch.nn as nn


from mmseg.models.builder import BACKBONES

from Allmodels.biformer.biformer import BiFormer
from timm.models.layers import LayerNorm2d

from Allmodels.biformer.checkpoint import load_checkpoint
from Allmodels.biformer.logger import get_root_logger


@BACKBONES.register_module()  
class BiFormer_mm(BiFormer):
    def __init__(self, pretrained=None, **kwargs):
        super().__init__(**kwargs)
        
        # step 1: remove unused segmentation head & norm
        del self.head # classification head
        del self.norm # head norm

        # step 2: add extra norms for dense tasks
        self.extra_norms = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))
        
        # Modified decoder head to output 512x512
        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim[-1], 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 16x16 -> 32x32
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 32x32 -> 64x64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 64x64 -> 128x128
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 128x128 -> 256x256
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 256x256 -> 512x512
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
        # step 3: initialization & load ckpt
        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            print(f'Load pretrained model from {pretrained}')

    def forward_features(self, x: torch.Tensor):
        out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out.append(self.extra_norms[i](x))
        return out[-1]  # Only return the last feature map

    def forward(self, x: torch.Tensor):
        features = self.forward_features(x)
        return self.decoder(features)  # Decode to match target size