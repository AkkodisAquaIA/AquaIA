import torch
import torch.nn as nn
from .position_encoding import build_position_encoding
from .DETR import DETR

class DINODetector(nn.Module):
    # TODO : Ajouter windowing strategy pour les images de grande taille
    # TODO : Ajouter TTA (test time augmentation) pour améliorer les performances (Soft-nms – improving object detection
    # with one line of code)
    def __init__(
        self, 
        img_size,
        backbone_id="dino", 
        detector_head_id="detr",
        positional_encoding_id="sine",
        d_model=256, # Transformer hidden dimension
        device="cpu",
        inference_mode=False,
        fp16=False,
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
        self.fp16 = fp16
        self.lora_ft = lora_ft
        self.quantize = quantize

        # Build model (backbone + positional encoding +  detection head)
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        # TODO : turn into a stateless function
        self.positional_encoding = build_position_encoding(positional_encoding_id, h_dim=d_model)
        self.detector = DETR(
            num_input_channels=self.backbone.embed_dim, 
            num_classes=num_classes, 
            num_queries=num_queries,
            d_model=d_model,
            fp16=fp16
        ).to(device)
        self.detector = torch.compile(self.detector, mode="max-autotune")
        self.patch_size = self.img_size // self.backbone.patch_size
        self.pe = self.positional_encoding(patch_x=self.patch_size, device=self.device) # precompute positional encoding for a single image (will be broadcasted in forward)
        if self.fp16:
            self.pe = self.pe.half()
        if not self.lora_ft:
            self.backbone.eval().to(device)

    def _forward_backbone(self, images):
        # TODO : check for useless permute and contiguous copy
        # Assuming square images for simplicity, otherwise we would need to compute patch_x and patch_y separately
        with torch.set_grad_enabled(self.lora_ft):
            features = self.backbone(images, is_training=True)["x_norm_patchtokens"] # (B, H*W, C)
            if self.fp16 and features.dtype != torch.float16:
                # The backbone may output float32 even under autocast, layernorm is the culprit ?
                features = features.half()
        return features, self.pe.unsqueeze(0).expand(features.shape[0], -1, -1) # (B, H*W, 2*num_pos_feats) add batch dimension with broadcasting

    def forward(self, images):
        with torch.inference_mode(self.device=="cpu" or self.inference_mode):
            # Dummy implementation of forward pass
            with torch.autocast(enabled=self.fp16, device_type="cuda", dtype=torch.float16):
                with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
                    # print(images.dtype)
                    embeddings, pe = self._forward_backbone(images)
                    out = self.detector(embeddings, pe)
        return out