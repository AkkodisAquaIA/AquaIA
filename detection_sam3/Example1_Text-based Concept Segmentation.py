from ultralytics.models.sam import SAM3SemanticPredictor

# Initialize predictor with configuration
overrides = dict(conf=0.25,
                 task="segment",
                 mode="predict",
                 model="C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/sam3.pt",
                 half=True,  # Use FP16 for faster inference
                 save=True,)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Set image once for multiple queries
predictor.set_image("C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/Person_bus_glasses.png")

# Query with multiple text prompts
results = predictor(text=["person", "bus", "glasses"])

# Works with descriptive phrases
#results = predictor(text=["person with red cloth", "person with blue cloth"])

# Query with a single concept
#results = predictor(text=["a person"])