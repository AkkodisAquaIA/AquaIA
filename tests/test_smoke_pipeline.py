"""Lightweight smoke tests on pure-Python helpers used by the pipeline.

We intentionally pick functions that:
  * do not touch the filesystem,
  * do not load model weights,
  * do not require a GPU.

If these break, something fundamental in the detection module is wrong.
This guards against a green CI that would otherwise only prove the imports
work but not that any actual code path runs.
"""

from __future__ import annotations


def test_infer_output_project_resolves_path_components() -> None:
    """`infer_output_project` joins task + model components into a path.

    Asserts the lowercase + underscore-join contract that
    `detection.utils.config_utils.infer_output_project` exposes — used by
    the training entry point to decide where to write artifacts.
    """
    from detection.utils.config_utils import infer_output_project

    config = {
        "training": {"task": "Detect"},
        "model": {"family": "DINOv3", "size": "Small", "init": "Pretrained"},
    }
    assert infer_output_project(config) == "results/detect/dinov3_small_pretrained"


def test_infer_output_project_omits_optional_parts() -> None:
    """Empty `size` / `init` must be skipped, not appear as trailing underscores."""
    from detection.utils.config_utils import infer_output_project

    config = {
        "training": {"task": "detect"},
        "model": {"family": "yolo"},
    }
    assert infer_output_project(config) == "results/detect/yolo"
