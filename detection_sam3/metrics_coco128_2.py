from ultralytics.models.sam import SAM3SemanticPredictor
from coco128_dict import COCO128_DICT

# Initialize predictor with configuration
overrides = dict(conf=0.25,
                 task="segment",
                 mode="predict",
                 model="C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/sam3.pt",
                 half=True,  # Use FP16 for faster inference
                 save=True,)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Set image once for multiple queries
predictor.set_image("C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/08_Data/coco128/images/train2017/000000000009.jpg")

# Query with multiple text prompts sourced from the COCO label dictionary (keys sorted for stable order)
text_prompts = [COCO128_DICT[idx] for idx in sorted(COCO128_DICT.keys())]
results = predictor(text=text_prompts)