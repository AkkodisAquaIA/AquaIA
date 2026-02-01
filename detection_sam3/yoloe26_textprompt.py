from pathlib import Path
from datetime import datetime
from ultralytics import YOLOE

# Initialize SAM3 predictor with configuration
current_folder = Path(__file__).resolve().parent
timestamp = datetime.now().strftime("%Y%m%d%H%M")

# Initialize a YOLOE model
model = YOLOE("C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/yoloe-26x-seg.pt")

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
names = ["person", "bus"]
model.set_classes(names, model.get_text_pe(names))

# Run detection on the given image
results = model.predict(
    source="C:/Users/zhijian.zhou/OneDrive - Akkodis/Bureau/unnamed.jpg",
    conf=0.25,
    half=True,
    save=True,
    imgsz=640,
    project=str(current_folder),
    name=f"yolo_result_det_{timestamp}",)

# Show results
results[0].show()