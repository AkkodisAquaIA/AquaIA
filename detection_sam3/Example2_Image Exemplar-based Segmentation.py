from ultralytics.models.sam import SAM3SemanticPredictor

# Initialize predictor
overrides = dict(conf=0.25,
                 task="segment",
                 mode="predict",
                 model="C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/sam3.pt",
                 half=True,
                 save=True)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Set image
predictor.set_image("C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/Person_bus_glasses.png")

# Provide bounding box examples to segment similar objects
#results = predictor(bboxes=[[480.0, 290.0, 590.0, 650.0]])

# Multiple bounding boxes for different concepts
results = predictor(bboxes=[[539, 599, 589, 639], [343, 267, 499, 662]])