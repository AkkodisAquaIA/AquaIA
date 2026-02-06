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
        backbone_id="dino", 
        detector_head_id="detr",
        positional_encoding_id="sine",
        d_model=256, # Transformer hidden dimension
        device=None,
        fp16=False,
        lora_ft=False,
        quantize=False,
        num_classes=91,
        num_queries=50,
        ):
        super(DINODetector, self).__init__()
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
        self.positional_encoding = build_position_encoding(positional_encoding_id, h_dim=d_model)
        self.detector = DETR(
            num_input_channels=self.backbone.embed_dim, 
            num_classes=num_classes, 
            num_queries=num_queries,
            d_model=d_model
        )
        print(self.detector)
        self.patch_size = self.backbone.patch_size
        if not self.lora_ft:
            self.backbone.eval()

    def _forward_backbone(self, images):
        # TODO : check for useless permute and contiguous copy
        # Assuming square images for simplicity, otherwise we would need to compute patch_x and patch_y separately
        patch_x = images.shape[2] // self.patch_size
        with torch.set_grad_enabled(self.lora_ft):
            features = self.backbone(images, is_training=True)["x_norm_patchtokens"]
            features = features.permute(0, 2, 1).contiguous()  # (B, C, H*W)
            features = features.view(features.shape[0], features.shape[1], patch_x, patch_x)  # (B, C, H, W)
        return features, self.positional_encoding(features) # (B, C, H, W)

    def forward(self, images):
        batch_size = images.shape[0]
        # Dummy implementation of forward pass
        embeddings, pe = self._forward_backbone(images)
        print("Embeddings shape:", embeddings.shape)
        print("PE shape:", pe.shape)
        out = self.detector(embeddings, pe)
        print(out["pred_logits"].shape)
        raise
        dummy_outputs = {
            "pred_logits": torch.randn(batch_size, self.num_queries, self.num_classes + 1),
            "pred_boxes": torch.sigmoid(torch.randn(batch_size, self.num_queries, 4)),
        }
        return dummy_outputs