from pathlib import Path

import torch

from detection.utils.plot_utils import save_sample_predictions


class _Dataset:
    dataset_root = Path("fake_dataset")
    data_split = "test"
    class_names = ["macroinvertebrate"]

    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {
            "input": torch.zeros(3, 8, 8),
            "image": torch.zeros(3, 8, 8),
            "img_path": f"image_{index}.png",
        }


class _Model:
    def eval(self):
        return self


def test_save_sample_predictions_respects_visualization_batch_size(tmp_path):
    observed_batch_sizes = []

    def predict_fn(model, samples, device, conf_thres):
        del model, device, conf_thres
        observed_batch_sizes.append(len(samples["img_paths"]))
        return [
            {
                "boxes": torch.empty((0, 4)),
                "scores": torch.empty((0,)),
                "labels": torch.empty((0,), dtype=torch.long),
            }
            for _ in samples["img_paths"]
        ]

    save_sample_predictions(
        model=_Model(),
        subset=_Dataset(),
        output_dir=tmp_path,
        predict_fn=predict_fn,
        num_samples=5,
        device="cpu",
        batch_size=2,
    )

    assert observed_batch_sizes == [2, 2, 1]
    assert len(list(tmp_path.glob("*.png"))) == 5
