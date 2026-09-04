import torch
import torch.nn as nn
from .position_encoding import build_position_encoding
from .DETR import DETR
from .backbone_id_map import resolve_backbone_id
# Z: torch.nn.attention tool box to control/expand attention mechanisms
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
        # Z: detector_head_id not used
        detector_head_id="detr",
        positional_encoding_id="sine",
        d_model=256,  # Transformer hidden dimension
        device="cpu",
        # Z: inference_mode not used
        inference_mode=False,
        # Z: lora_ft not used
        lora_ft=False,
        # Z: quantize not used
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
        # Z: torch.hub.load() can load models from local or online repo
        # Z: **dict unpacks the dictionary returned into keyword arguments for exterior function call
        self.backbone = torch.hub.load(**resolve_backbone_id(backbone_id))
        # TODO : turn into a stateless function
        self.positional_encoding = build_position_encoding(positional_encoding_id, h_dim=d_model)
        self.detector = DETR(num_input_channels=self.backbone.embed_dim, num_classes=num_classes, num_queries=num_queries, d_model=d_model).to(device)
        # Assuming square images for simplicity, otherwise we would need to compute patch_x and patch_y separately
        # Z: here is the patch grid size
        self.patch_size = self.img_size // self.backbone.patch_size

        # precompute positional encoding once for a single image (will be broadcasted in forward)
        # Z: assuming equal size for all images
        self.pe = self.positional_encoding(patch_x=self.patch_size, device=self.device)

        if not self.lora_ft:
            # Z: only backbone
            self.backbone.eval().to(device)
        else:
            raise NotImplementedError("LoRA fine-tuning not implemented yet, set lora_ft to False for now")

    def _forward_backbone(self, inputs):
        # Feed input to backbone and extract features
        # Z: lora_ft = True -> need grad, lora_ft = False -> no grad
        with torch.set_grad_enabled(self.lora_ft):
            # Z: if lora_ft = False, backbone not trained
            # Z: Return outputs following the DINOv3's training path, richer results
            features = self.backbone(inputs, is_training=True)["x_norm_patchtokens"]  # (B, H*W, C)
        # Z: expand PE to batch size
        return features, self.pe.unsqueeze(0).expand(features.shape[0], -1, -1)  # (B, H*W, 2*num_pos_feats) add batch dimension with broadcasting

    def forward(self, inputs):
        if inputs.device.type == "cuda":
            # Z: if GPU then flash attention or efficient attention else normal math attention
            # Z: the choice depends on pytorch itself
            backends = [attn.SDPBackend.FLASH_ATTENTION, attn.SDPBackend.EFFICIENT_ATTENTION]
        else:
            backends = [attn.SDPBackend.MATH]
        with torch.nn.attention.sdpa_kernel(backends=backends):
            embeddings, pe = self._forward_backbone(inputs)
            out = self.detector(embeddings, pe)
        # Z: out = dict with keys "pred_logits", "pred_boxes", "aux_outputs" (if aux_loss=True)
        return out
