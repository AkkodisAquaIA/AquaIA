from pathlib import Path

from ultralytics.models.sam import SAM3SemanticPredictor

from coco128_dict import COCO128_DICT

MODEL_PATH = "C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/sam3.pt"
IMAGES_FOLDER = "C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/08_Data/coco128/images/train2017"
SAVE_FOLDER = "C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/detection_sam3/metrics_coco128_2_results"

# Initialize predictor with configuration
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model=MODEL_PATH,
    half=True,  # Use FP16 for faster inference
    save=True,
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Text prompts sourced from the COCO label dictionary (keys sorted for stable order)
text_prompts = [COCO128_DICT[idx] for idx in sorted(COCO128_DICT.keys())]

# Iterate over all images in IMAGES_FOLDER and run inference with the text prompts
image_files = sorted(f for f in Path(IMAGES_FOLDER).glob("**/*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"})

for img in image_files:
    predictor.set_image(str(img))
    results = predictor(text=text_prompts)