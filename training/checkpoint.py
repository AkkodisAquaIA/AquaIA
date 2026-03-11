import torch


def get_model_state_dict(model, clone_to_cpu=False):
    # Model compilation add some additional attr, take original model for saving
    state_dict_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    state_dict = state_dict_model.state_dict()
    if not clone_to_cpu:
        return state_dict
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def update_best_checkpoint(best_loss, best_epoch, best_model_state_dict, epoch, epoch_metrics, model):
    if epoch_metrics["avg"] >= best_loss:
        return best_loss, best_epoch, best_model_state_dict

    return (
        epoch_metrics["avg"],
        epoch + 1,
        get_model_state_dict(model, clone_to_cpu=True),
    )


def save_model_checkpoint(
    path,
    model_state_dict,
):
    checkpoint = {"model_state_dict": model_state_dict}
    torch.save(checkpoint, path)


def save_training_state_checkpoint(
    path,
    epoch,
    optimizer_state_dict,
    scaler_state_dict,
    scheduler_state_dict=None,
):
    checkpoint = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer_state_dict,
        "scaler_state_dict": scaler_state_dict,
    }
    if scheduler_state_dict is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state_dict
    torch.save(checkpoint, path)
