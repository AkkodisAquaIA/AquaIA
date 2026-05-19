# Tests

Smoke-level pytest suite. No GPU required — all tests run on CPU.

## Run

```bash
pytest                        # full suite (quiet)
pytest -v                     # verbose
pytest tests/test_imports.py  # fastest sanity check
```

## Coverage

| File | What it checks |
|---|---|
| `test_imports.py` | Top-level packages import without error |
| `test_main.py` | `python main.py --help` exits 0 and lists `train` / `infer` |
| `test_smoke_pipeline.py` | `infer_output_project` pure-Python logic (no model, no GPU) |

## CI

The `test` job in [`ci.yml`](../.github/workflows/ci.yml) installs CPU torch first, then runs `pytest -q`. A green local run is a strong signal the PR will pass CI.

## What's missing

Heavier model-level evaluation (accuracy, mAP, confusion matrices) is part of **Task 2.3** of the scientific dossier and will be added as the pipeline matures.
