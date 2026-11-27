# PyTorch Profiler is a tool that allows the collection of execution performance
# during training and inference
from torch.profiler import profile, ProfilerActivity, schedule


def torch_profiler_factory(wait=1, warmup=1, active=3):
    """Create a PyTorch profiler with specified wait, warmup, and active steps."""
    torch_profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            # No recording for the first "wait" steps
            wait=wait,
            # Warmup for "warmup" steps
            warmup=warmup,
            # Record for "active" steps
            active=active,
        ),
    )
    return torch_profiler
