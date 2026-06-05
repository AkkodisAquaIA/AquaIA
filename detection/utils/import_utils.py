import torch

try:
    import nvidia.dali as dali
    import nvidia.dali.experimental.dynamic as ndd # noqa: F401
    from nvidia.dali.plugin.pytorch import DALIRaggedIterator, LastBatchPolicy # noqa: F401

    # Import DALI components to avoid having to import them in every file that uses DALI
    fn = dali.fn
    pipeline_def = dali.pipeline_def
    types = dali.types
    if not torch.cuda.is_available():
        DALI_AVAILABLE = False
    else:
        DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
