import torch
import torch.nn as nn
from .position_encoding import build_position_encoding
from .DETR import DETR
from .backbone_id_map import resolve_backbone_id
import torch.nn.attention as attn

class DINODetector(nn.Module):
    # TODO : Ajouter windowing strategy pour les images de grande taille
    # TODO : Ajouter TTA (test time augmentation) pour améliorer les performances (Soft-nms – improving object detection
    # with one line of code)
    # TODO : tester transfomers with dynamic tanh (no layer norms)
    # TODO : tester MALA (Magnitude Aware Linear Attention)
    def __init__(
        self, 
        img_size,
        backbone_id="dinov3_small", 
        detector_head_id="detr",
        positional_encoding_id="sine",
        d_model=256, # Transformer hidden dimension
        device="cpu",
        inference_mode=False,
        lora_ft=False,
        quantize=False,
        num_classes=91,
        num_queries=50,
        ):
        super(DINODetector, self).__init__()
        self.img_size = img_size
        self.inference_mode = inference_mode
        self.backbone_id = backbone_id
        self.detector_head_id = detector_head_id
        self.device = device
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.lora_ft = lora_ft
        self.quantize = quantize

        # ----- Build model (backbone + positional encoding +  detection head) -----
        self.backbone = torch.hub.load(**resolve_backbone_id(backbone_id))
        # TODO : turn into a stateless function
        self.positional_encoding = build_position_encoding(positional_encoding_id, h_dim=d_model)
        self.detector = DETR(
            num_input_channels=self.backbone.embed_dim, 
            num_classes=num_classes, 
            num_queries=num_queries,
            d_model=d_model
        ).to(device)
        # Assuming square images for simplicity, otherwise we would need to compute patch_x and patch_y separately
        self.patch_size = self.img_size // self.backbone.patch_size

        # precompute positional encoding once for a single image (will be broadcasted in forward) 
        self.pe = self.positional_encoding(patch_x=self.patch_size, device=self.device) 

        if not self.lora_ft:
            self.backbone.eval().to(device)
        else:
            raise NotImplementedError("LoRA fine-tuning not implemented yet, set lora_ft to False for now")

    def _forward_backbone(self, images):
        # Feed input to backbone and extract features
        with torch.set_grad_enabled(self.lora_ft):
            features = self.backbone(images, is_training=True)["x_norm_patchtokens"] # (B, H*W, C)
        return features, self.pe.unsqueeze(0).expand(features.shape[0], -1, -1) # (B, H*W, 2*num_pos_feats) add batch dimension with broadcasting

    def forward(self, images):
        if images.device.type == "cuda":
            backends = [attn.SDPBackend.FLASH_ATTENTION, attn.SDPBackend.EFFICIENT_ATTENTION]
        else:
            backends = [attn.SDPBackend.MATH]
        with torch.nn.attention.sdpa_kernel(backends=backends):
            embeddings, pe = self._forward_backbone(images)
            out = self.detector(embeddings, pe)
        return out
