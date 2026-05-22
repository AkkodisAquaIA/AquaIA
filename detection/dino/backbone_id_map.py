import os


# TODO : temporaire, DINOv3 sera téléchargé depuis le sharepoint idéalement
default_dir = os.path.expanduser("~/.cache")
ROOT_DIR = os.environ.get("AQUAIA_MODEL_ROOT_DIR", default_dir)
DINOV2_LOCAL_REPO = os.path.join(ROOT_DIR+"/torch/hub", "facebookresearch_dinov2_main")
DINOV3_LOCAL_REPO = os.path.join(ROOT_DIR+"/torch/hub", "facebookresearch_dinov3_main")

DINO_ID_MAPPING = {
	"dinov2_small" : 
		{
			"repo_or_dir" : DINOV2_LOCAL_REPO, 
			"model" :  "dinov2_vits14_reg",
			"source" : "local",
		},
	"dinov2_base" : 
		{
			"repo_or_dir" : DINOV2_LOCAL_REPO, 
			"model" :  "dinov2_vitb14_reg",
			"source" : "local",
		},
	"dinov2_large" : 
		{
			"repo_or_dir" : DINOV2_LOCAL_REPO, 
			"model" :  "dinov2_vitl14_reg",
			"source" : "local",
		},
	"dinov3_small" : 
		{
			"repo_or_dir" : DINOV3_LOCAL_REPO, 
			"model" :  "dinov3_vits16",
			"source" : "local",
			"weights" : os.path.join(ROOT_DIR, "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
		},
	"dinov3_plus" : 
		{
			"repo_or_dir" : DINOV3_LOCAL_REPO, 
			"model" :  "dinov3_vits16plus",
			"source" : "local",
			"weights" : os.path.join(ROOT_DIR, "checkpoints/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth")
		},
	"dinov3_base" : 
		{
			"repo_or_dir" : DINOV3_LOCAL_REPO, 
			"model" :  "dinov3_vitb16",
			"source" : "local",
			"weights" : os.path.join(ROOT_DIR, "checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
		},
	"dinov3_large" : 
		{
			"repo_or_dir" : DINOV3_LOCAL_REPO, 
			"model" :  "dinov3_vitl16",
			"source" : "local",
			"weights" : os.path.join(ROOT_DIR, "checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
		},
}

def resolve_backbone_id(backbone_id):
	if backbone_id not in DINO_ID_MAPPING:
		raise ValueError(f"Backbone id {backbone_id} not found in mapping")
	dino_backbone_info = DINO_ID_MAPPING[backbone_id]
	local_repo = dino_backbone_info["repo_or_dir"]
	if not os.path.exists(local_repo):
		raise ValueError(f"Le dossier contenant les modèles DINO n'existe pas : {local_repo}.")
	return dino_backbone_info
