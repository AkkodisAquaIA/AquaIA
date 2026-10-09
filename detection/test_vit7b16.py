"""Standalone offline DINOv3-7B COCO detector CUDA memory smoke test.

The image contains the official repository at detection/vit7b16_assets/dinov3/.
Weights are stored on the VM under /aqua-ia-data/models/checkpoints/vit7b16/:
  dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
  dinov3_vit7b16_coco2017_detr_head-b0235ff7.pth
Mount /aqua-ia-data into the container at the same path to use these defaults.
Use --repo-dir, --backbone-weights and --detector-weights to override paths.
Use --batch-size to test larger batches; a final partial batch is kept.
Only local files are loaded. No AquaIA training modules are imported.
Exit codes: 0 = all forwards passed, 2 = CUDA OOM, 1 = another error.
This is a resized-image FP16 smoke test, not a COCO accuracy benchmark.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import sys
import time
import traceback
from urllib.parse import urlparse
from urllib.request import url2pathname


ASSETS = Path(__file__).resolve().parent / "vit7b16_assets"
WEIGHTS_DIR = Path("/aqua-ia-data/models/checkpoints/vit7b16")


def parse_args(argv=None):
    "Define command-line arguments."
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=ASSETS / "dinov3")
    parser.add_argument(
        "--backbone-weights", type=Path,
        default=WEIGHTS_DIR / "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    )
    parser.add_argument(
        "--detector-weights", type=Path,
        default=WEIGHTS_DIR / "dinov3_vit7b16_coco2017_detr_head-b0235ff7.pth",
    )
    parser.add_argument("--image-dir", type=Path, default=Path(
        "/aqua-ia-data/datasets/detection/coco_custom_match/images/train"))
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1, help="Images per forward; the last batch may be smaller")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=640, help="Square resized input")
    parser.add_argument("--conf", type=float, default=0.3, help="Output filter only; does not reduce forward memory")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/results/vit7b16_smoke"))
    args = parser.parse_args(argv)
    if args.imgsz % 16:
        parser.error("imgsz must be a multiple of 16")
    return args


def select_images(image_dir, num_images, seed):
    """Select a reproducible random sample of images from the directory."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = sorted(p for p in image_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) < num_images:
        raise ValueError(f"Need {num_images} images, found {len(images)} under {image_dir}")
    return sorted(random.Random(seed).sample(images, num_images))


def cuda_memory(torch):
    """Return a dict of current and peak CUDA memory usage in GiB, plus device free/total."""
    free, total = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    return {
        "allocated_gib": torch.cuda.memory_allocated() / gib,
        "reserved_gib": torch.cuda.memory_reserved() / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / gib,
        "device_free_gib": free / gib,
        "device_total_gib": total / gib,
    }


def save_report(path, report):
    # Flush after each stage/batch so completed results survive a later OOM.
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


@contextmanager
def local_fp16_loading(torch):
    """Create a temporary load env, build model with FP16 in with block,
    use memory mapping to load weights, restore original loader and dtype on exit.
    """
    original_loader = torch.hub.load_state_dict_from_url
    original_dtype = torch.get_default_dtype()

    def load_checkpoint(url, **kwargs):
        parsed = urlparse(url)
        if parsed.scheme != "file" or parsed.netloc:
            raise ValueError(f"Only local checkpoint files are allowed: {url}")
        path = Path(url2pathname(parsed.path))
        print(f"Loading local checkpoint: {path}", flush=True)
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)

    try:
        torch.set_default_dtype(torch.float16)
        torch.hub.load_state_dict_from_url = load_checkpoint
        yield
    finally:
        torch.hub.load_state_dict_from_url = original_loader
        torch.set_default_dtype(original_dtype)


def run(args):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "summary.json"
    report = {"status": "running", "stage": "preflight", "batch_size": args.batch_size,
              "precision": "fp16", "imgsz": args.imgsz, "conf": args.conf,
              "seed": args.seed, "batches": [], "output_dir": str(output_dir)}
    save_report(report_path, report)
    print(f"Report: {report_path}", flush=True)
    # later if torch is not None to check if pytoch imported successfully
    torch = None
    try:
        # ====== Initialize dino3 repo, images, weights, report metadata======
        repo_dir = args.repo_dir.resolve()
        for path in (repo_dir / "dinov3/hub/detectors.py",
                     args.backbone_weights, args.detector_weights):
            if not path.is_file():
                raise FileNotFoundError(f"Required local asset not found: {path}")
        selected = select_images(args.image_dir, args.num_images, args.seed)
        report["selected_images"] = [str(p) for p in selected]
        report["repo_dir"] = str(repo_dir)
        report["backbone_weights"] = str(args.backbone_weights.resolve())
        report["detector_weights"] = str(args.detector_weights.resolve())

        # ====== Import torch and check CUDA ======
        import torch
        from PIL import Image
        from torchvision.transforms import v2
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable!")
        report["gpu"] = torch.cuda.get_device_name()
        report["torch_version"] = str(torch.__version__)
        report["cuda_version"] = torch.version.cuda
        report["initial_memory"] = cuda_memory(torch)
        print(f"GPU: {report['gpu']} | torch {torch.__version__} | CUDA {torch.version.cuda}", flush=True)

        # ====== Load the official DINOv3-7B detector in FP16 on CPU ======
        # Add repo in first place
        sys.path.insert(0, str(repo_dir))
        from dinov3.hub.detectors import dinov3_vit7b16_de
        report["stage"] = "load_model_cpu"
        save_report(report_path, report)
        print("Constructing the full official detector on CPU in FP16...", flush=True)
        started = time.perf_counter()
        with local_fp16_loading(torch):
            model = dinov3_vit7b16_de(
                weights=str(args.detector_weights.resolve()),
                backbone_weights=str(args.backbone_weights.resolve()),
            )
        # Set to eval and disable gradients
        model.eval().requires_grad_(False)
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        report["parameter_gib"] = parameter_bytes / 1024 ** 3
        report["parameter_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
        if report["parameter_dtypes"] != ["torch.float16"]:
            raise RuntimeError(f"Expected FP16 parameters, got {report['parameter_dtypes']}")
        # Exacute Python garbage collection
        gc.collect()
        report["cpu_load_seconds"] = time.perf_counter() - started

        # ====== Move model to CUDA ======
        report["stage"] = "move_model_to_cuda"
        save_report(report_path, report)
        torch.cuda.reset_peak_memory_stats()
        print(f"Moving model to CUDA; parameters: {report['parameter_gib']:.2f} GiB", flush=True)
        model = model.to(device="cuda")
        # Wait for all CUDA kernels to finish
        torch.cuda.synchronize()
        report["model_memory"] = cuda_memory(torch)
        save_report(report_path, report)
        print(f"Model memory: {report['model_memory']}", flush=True)

        # ====== Forward images in batches, in FP16 ======
        # Build a torchvision v2 transform to resize and normalize the image for model
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((args.imgsz, args.imgsz), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        num_batches = (len(selected) + args.batch_size - 1) // args.batch_size

        # For each batch
        for batch_index, start in enumerate(range(0, len(selected), args.batch_size), start=1):
            paths = selected[start:start + args.batch_size]
            report["stage"] = f"forward_batch_{batch_index}"
            report["current_images"] = [str(path) for path in paths]
            report["actual_batch_size"] = len(paths)
            save_report(report_path, report)
            tensors = []
            for path in paths:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                tensors = [tensor.to(device="cuda", dtype=torch.float16) for tensor in tensors]
                predictions = model(tensors)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            memory = cuda_memory(torch)

            # For each image
            for offset, (path, prediction) in enumerate(zip(paths, predictions)):
                index = start + offset + 1
                # Check boxes and scores for non-finite values
                for key in ("boxes", "scores"):
                    if not torch.isfinite(prediction[key]).all().item():
                        raise RuntimeError(f"Non-finite {key} in prediction for {path}")
                # Filter predictions by confidence threshold
                keep = prediction["scores"] >= args.conf
                filtered = {key: value[keep].detach().float().cpu().tolist()
                            for key, value in prediction.items()}
                filtered["labels"] = [int(value) for value in filtered["labels"]]
                # Per image JSON result file
                result = {"image": str(path),
                          "batch_index": batch_index,
                          "predictions": filtered}
                (output_dir / f"prediction_{index}.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8")

            report["batches"].append({"batch_index": batch_index, "batch_size": len(paths),
                                      "images": [str(path) for path in paths],
                                      "seconds": elapsed, "memory": memory})
            save_report(report_path, report)
            print(f"[batch {batch_index}/{num_batches}] size={len(paths)}: PASS, {elapsed:.2f}s, "
                  f"peak allocated={memory['peak_allocated_gib']:.2f} GiB, "
                  f"peak reserved={memory['peak_reserved_gib']:.2f} GiB", flush=True)
            del tensors, predictions, prediction, keep
            gc.collect()
            torch.cuda.empty_cache()

        report["status"] = "passed"
        report["stage"] = "complete"
        report.pop("current_images", None)
        report.pop("actual_batch_size", None)
        save_report(report_path, report)
        print(f"PASS: {len(selected)} images, batch={args.batch_size}, FP16, imgsz={args.imgsz}", flush=True)
        return 0

    # If OOM or another error occurs, save the report and exit with a non-zero code.
    except Exception as exc:
        is_cuda_oom = torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError)
        report["status"] = "cuda_oom" if is_cuda_oom else "error"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        save_report(report_path, report)
        print(f"{report['status'].upper()} during {report['stage']}: {exc}", flush=True)
        traceback.print_exc()
        return 2 if is_cuda_oom else 1


if __name__ == "__main__":
    sys.exit(run(parse_args()))
