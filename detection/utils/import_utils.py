import torch

try:
    import nvidia.dali as dali

    # Import DALI components to avoid having to import them in every file that uses DALI
    fn = dali.fn
    pipeline_def = dali.pipeline_def
    types = dali.types
    ndd = dali.experimental.dynamic
    DALIRaggedIterator = dali.plugin.pytorch.DALIRaggedIterator
    LastBatchPolicy = dali.plugin.pytorch.LastBatchPolicy
    if not torch.cuda.is_available():
        DALI_AVAILABLE = False
    else:
        DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
