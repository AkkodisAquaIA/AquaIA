# Images to be predicted with detection models
IMAGES_FOLDER = "C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/08_Data/coco128/images/train2017"

# Active model key: "sam3" or "yoloe26"
MODEL_NAME = "sam3"

MODEL_CFG = {
    "sam3": {
        "CONF": 0.5,
        "TASK": "segment",
        "MODE": "predict",
        "PATH": "C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/sam3.pt",
        "HALF": True,   # Use FP16 for faster inference
        "SAVE": True,   # Save results to project folder
        "IMGSZ": 640,
        "NMS": False,   # Not original NMS
    },
    "yoloe26": {
        "CONF": 0.3,
        "PATH": "C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/yoloe-26x-seg.pt",
        "HALF": False,
        "SAVE": True,
        "IMGSZ": 640,
        "NMS": False,   # Not original NMS
    },
}