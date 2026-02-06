from Detection.DINO.dino_detector import DINODetector
from Detection.loss import SetCriterion
from Detection.utils.matcher import HungarianMatcher
import torch
import glob
import os

def _parse_label_line(line):
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Invalid label line: {line}")
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])
    return class_id, [x_center, y_center, width, height]

def load_targets(data_path):
    targets_path = f"{data_path}/labels/"
    targets_files = glob.glob(os.path.join(targets_path, "*.txt"))
    targets = []
    for target_file in targets_files:
        with open(target_file, 'r') as f:
            label_data = f.read().strip().splitlines()
            img_lbls = []
            img_bboxs = []
            for line in label_data:
                class_id, bbox = _parse_label_line(line)
                img_lbls.append(class_id)
                img_bboxs.append(bbox)
            target = {
                "labels": torch.tensor(img_lbls, dtype=torch.int64),
                "boxes": torch.tensor(img_bboxs, dtype=torch.float32),
            }
            targets.append(target)
    return targets
                
matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, cost_bbox_type='l1')
loss_weight_dict = {
    "loss_ce": 2,
    "loss_bbox": 5,
    "loss_giou": 2
}
criterion = SetCriterion(num_classes=91, matcher=matcher, weight_dict=loss_weight_dict)
dummy_images = torch.randn(128, 3, 224, 224)  # Batch of 2 images
dataset_name = "coco128"
dataset_dir = "data"
data_path = os.path.join(dataset_dir, dataset_name)
targets = load_targets(data_path)
model = DINODetector()
y = model(dummy_images)

loss_dict = criterion(y, targets)
print(loss_dict)