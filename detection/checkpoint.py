import torch


def load_pretrain_model_checkpoint(path, model):
    """Load pretrained checkpoint weights to a built model, keeping the new class layer when the class count changes."""
    checkpoint = torch.load(path, map_location="cpu")
    # Checkpoint weights
    state_dict = checkpoint["model_state_dict"]
    # Newly built model param dict
    model_state_dict = model.state_dict()
    # Classification layer weights and bias keys
    class_keys = {"detector.class_embed.weight", "detector.class_embed.bias"}
    # If num_classes in the checkpoint is different from newly built model
    reset_class_head = state_dict["detector.class_embed.weight"].shape[0] != model.num_classes
    # Newly built model param keys
    expected_keys = set(model_state_dict)
    # Check if checkpoint keys match newly built model keys
    if set(state_dict) != expected_keys:
        raise ValueError("Initial checkpoint does not match the model parameter names")
    for key in expected_keys:
        # If num_classes changes and key in classification layer, skip
        if reset_class_head and key in class_keys:
            continue
        # Check if checkpoint param shape matches newly built model param shape
        if state_dict[key].shape != model_state_dict[key].shape:
            raise ValueError(f"Initial checkpoint shape mismatch for {key}: {state_dict[key].shape} != {model_state_dict[key].shape}")
    # If num_classes change
    if reset_class_head:
        state_dict = {key: value for key, value in state_dict.items() if key not in class_keys}
        # Load checkpoint weights to model, skipping classification layer
        model.load_state_dict(state_dict, strict=False)
    # If num_classes do not change, load all checkpoint weights to model
    else:
        model.load_state_dict(state_dict)
    return reset_class_head


def _get_model_state_dict(model, clone_to_cpu=False):
    """Extract a savable state_dict from model that is compatible with models using torch.compile()."""
    # Model compilation add some additional attr, take original model for saving
    state_dict_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    # Get model params dict
    state_dict = state_dict_model.state_dict()
    if not clone_to_cpu:
        return state_dict
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def save_model_checkpoint(path, model):
    """Save model state dict for training and resume."""
    model_state_dict = _get_model_state_dict(model, clone_to_cpu=True)
    checkpoint = {"model_state_dict": model_state_dict}
    torch.save(checkpoint, path)


def save_training_state_checkpoint(path, epoch, optimizer_state_dict, scaler_state_dict, scheduler_state_dict=None, early_stopping_state=None):
    # Scaler state dict is needed for mixed precision training
    """Save model epoch, optimizer state dict, scaler state dict, early_stopping_state
    and optionally scheduler state dict. For training and resume."""
    checkpoint = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer_state_dict,
        "scaler_state_dict": scaler_state_dict,
    }
    if scheduler_state_dict is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state_dict
    checkpoint["early_stopping_state"] = early_stopping_state
    torch.save(checkpoint, path)


def load_training_state_checkpoint(path, device="cpu"):
    """Load model epoch, optimizer state dict, scaler state dict and optionally scheduler state dict.
    For training and resume."""
    return torch.load(path, map_location=device)
