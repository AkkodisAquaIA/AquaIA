from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/yoloe-26x-seg.pt")  # or yoloe-26s/m-seg.pt for different sizes

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
names = ["person", "bus"]
model.set_classes(names, model.get_text_pe(names))

# Run detection on the given image
results = model.predict("C:/Users/zhijian.zhou/OneDrive - Akkodis/Bureau/unnamed.jpg")

# Show results
results[0].show()