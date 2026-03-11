from pathlib import Path

import torch
import tqdm

from Detection.DINO.dino_detector import DINODetector
from test.config_utils import find_latest_run_dir, load_class_names, load_run_config
from test.dataset_utils import sample_dataset
from test.plot_utils import annotate_images_with_predictions


def load_best_model(run_dir, run_config):
    run_dir = Path(run_dir)
    if run_config is None:
        raise ValueError("resolved_config.yaml is required to load the model metadata.")
    img_size = run_config["data"].get("img_size")
    num_classes = run_config["data"].get("num_classes")
    if img_size is None or num_classes is None:
        raise ValueError("resolved_config.yaml is missing `data.img_size` or `data.num_classes`.")

    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu")

    model = DINODetector(
        img_size=int(img_size),
        device="cpu",
        num_classes=int(num_classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_for_images(model, images):
    outputs = model(images)
    outputs["pred_boxes"] = outputs["pred_boxes"].float()
    outputs["pred_logits"] = outputs["pred_logits"].float()
    return outputs


def run_inference_for_split(model, split_name, dataset_root, class_names, inference_config, output_root, seed):
    images, image_ids = sample_dataset(
        dataset_root=dataset_root,
        num_samples=inference_config["num_samples"],
        seed=seed,
    )
    split_output_dir = output_root / f"{split_name}_predictions"

    print(f"{split_name}: sampled {len(image_ids)} images from {dataset_root}")
    for start in tqdm.tqdm(range(0, len(image_ids), inference_config["batch"]), desc=f"Testing {split_name}"):
        end = min(start + inference_config["batch"], len(image_ids))
        batch_images = images[start:end]
        batch_image_ids = image_ids[start:end]
        outputs = predict_for_images(model=model, images=batch_images)
        annotate_images_with_predictions(
            images=batch_images,
            outputs=outputs,
            class_names=class_names,
            conf_thres=inference_config["conf"],
            output_dir=split_output_dir,
            image_ids=batch_image_ids,
        )


def test_dino(config):
    run_cfg = config["run"]
    data_config = config["data"]
    inference_config = config["inference"]
    output_config = config["output"]

    run_dir = Path(run_cfg["run_dir"]) if run_cfg.get("run_dir") else find_latest_run_dir(run_cfg["runs_root"])
    run_config = load_run_config(run_dir)
    train_data_root = run_config["data"]["dataset_yaml"]
    test_data_root = data_config.get("test_data_root")
    model = load_best_model(run_dir, run_config=run_config)
    class_names = load_class_names(test_data_root)

    output_dir = Path(output_config["output_dir"]) if output_config.get("output_dir") else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating run: {run_dir}")
    print(f"Train dataset: {train_data_root}")
    print(f"Test dataset: {test_data_root}")
    print(f"Saving predictions under: {output_dir}")

    run_inference_for_split(
        model=model,
        split_name="train",
        dataset_root=train_data_root,
        class_names=class_names,
        inference_config=inference_config,
        output_root=output_dir,
        seed=inference_config["seed"],
    )
    run_inference_for_split(
        model=model,
        split_name="test",
        dataset_root=test_data_root,
        class_names=class_names,
        inference_config=inference_config,
        output_root=output_dir,
        seed=inference_config["seed"] + 1,
    )
    return run_dir
