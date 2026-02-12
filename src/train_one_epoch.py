import time
import torch
from torch import nn
from . import utils


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    args,
    model_ema=None,
    scaler=None,
    writer=None,
    progress=None,
    log_layout=None,
    epoch_task_id=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.3g}"))
    metric_logger.add_meter(
        "img/s", utils.SmoothedValue(window_size=10, fmt="{value:.1f}")
    )

    header = f"Epoch: [{epoch + 1}]"
    for i, (image, target) in enumerate(
        metric_logger.log_every(
            data_loader, args.print_freq, header, progress, log_layout
        )
    ):
        start_time = time.time()
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            loss = criterion(model(image)[0], target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                clip_target = getattr(model, "module", model)
                nn.utils.clip_grad_norm_(
                    clip_target.parameters(), args.clip_grad_norm
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                model_ema.n_averaged.fill_(0)

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["img/s"].update(
            image.shape[0] / (time.time() - start_time)
        )
        if writer:
            writer.add_scalar("Loss/train", loss.item(), epoch * len(data_loader) + i)

        if progress and epoch_task_id is not None:
            progress.update(epoch_task_id, advance=1.0 / len(data_loader))

    return metric_logger
