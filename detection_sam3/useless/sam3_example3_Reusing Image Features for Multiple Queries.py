import cv2

from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics.utils.plotting import Annotator, colors
# Z: Annotator is used for visualizing masks on images
# Z: colors is used to generate distinct colors for different masks

# Initialize predictors
overrides = dict(conf=0.50,
                 task="segment",
                 mode="predict",
                 model="C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/sam3.pt",
                 verbose=False)
predictor = SAM3SemanticPredictor(overrides=overrides)
predictor2 = SAM3SemanticPredictor(overrides=overrides)

# Extract features from the first predictor
source = "C:/Users/zhijian.zhou/OneDrive - Akkodis/Travail/10_AquaIA/01_Git/Person_bus_glasses.png"
predictor.set_image(source) # Z: set_image extracts and stores image features internally
src_shape = cv2.imread(source).shape[:2] # Z: (height, width)

# Setup second predictor and reuse features
predictor2.setup_model() # Z: setup_model initializes the model without extracting features

# Perform inference using shared features with text prompt
masks, boxes = predictor2.inference_features(predictor.features, src_shape=src_shape, text=["person"])

# Perform inference using shared features with bounding box prompt
#masks, boxes = predictor2.inference_features(predictor.features, src_shape=src_shape, bboxes=[[439, 437, 524, 709]])

# Visualize results
if masks is not None:
    masks, boxes = masks.cpu().numpy(), boxes.cpu().numpy()
    im = cv2.imread(source)
    annotator = Annotator(im, pil=False)
    annotator.masks(masks, [colors(x, True) for x in range(len(masks))])

    cv2.imshow("result", annotator.result())
    cv2.waitKey(0)  # Z: Wait for a key press to close the image window