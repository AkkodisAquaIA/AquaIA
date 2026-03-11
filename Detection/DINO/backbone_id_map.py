import os


# TODO : temporaire, DINOv3 sera téléchargé depuis le sharepoint idéalement
ROOT_DIR = os.path.expanduser("~/.cache/torch/hub")
DINOV3_LOCAL_PATH = os.path.join(ROOT_DIR, "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
DINOV3_LOCAL_REPO = os.path.join(ROOT_DIR, "facebookresearch_dinov3_main")

DINO_ID_MAPPING = {
	"v2_small" : 
		{
			"repo_or_dir" : "facebookresearch/dinov2", 
			"model" :  "dinov2_vits14_reg"
		},
	"v3_small" : 
		{
			"repo_or_dir" : DINOV3_LOCAL_REPO, 
			"model" :  "dinov3_vits16",
			"source" : "local",
			"weights" : DINOV3_LOCAL_PATH
		},
}

def resolve_backbone_id(backbone_id):
	if backbone_id not in DINO_ID_MAPPING:
		raise ValueError(f"Backbone id {backbone_id} not found in mapping")
	return DINO_ID_MAPPING[backbone_id]