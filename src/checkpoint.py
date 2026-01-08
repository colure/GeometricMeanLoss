import os
from . import utils


def save_checkpoint(
    model_without_ddp,
    optimizer,
    lr_scheduler,
    epoch,
    args,
    output_dir,
    filename="checkpoint.pth",
    model_ema=None,
    scaler=None,
):
    checkpoint = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "args": args,
    }

    if model_ema:
        checkpoint["model_ema"] = model_ema.state_dict()

    if scaler:
        checkpoint["scaler"] = scaler.state_dict()

    save_path = os.path.join(output_dir, "checkpoints", filename)
    utils.save_on_master(checkpoint, save_path)

    return save_path


def save_best_checkpoint(
    model_without_ddp,
    optimizer,
    lr_scheduler,
    epoch,
    args,
    output_dir,
    shot,
    model_ema=None,
    scaler=None,
    is_ema=False,
):
    suffix = "_ema" if is_ema else ""
    filename = f"checkpoint_best_shot{shot}{suffix}.pth"

    return save_checkpoint(
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epoch=epoch,
        args=args,
        output_dir=output_dir,
        filename=filename,
        model_ema=model_ema,
        scaler=scaler,
    )


def save_final_checkpoint(
    model_without_ddp,
    optimizer,
    lr_scheduler,
    epoch,
    args,
    output_dir,
    model_ema=None,
    scaler=None,
):
    return save_checkpoint(
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epoch=epoch,
        args=args,
        output_dir=output_dir,
        filename="checkpoint_final.pth",
        model_ema=model_ema,
        scaler=scaler,
    )
