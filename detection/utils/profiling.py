from torch.profiler import profile, ProfilerActivity, schedule


def torch_profiler_factory(wait=1, warmup=1, active=3):
    torch_profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=wait,
            warmup=warmup,
            active=active,
        ),
    )
    return torch_profiler
