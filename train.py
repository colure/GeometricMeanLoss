import datetime
import os
import time

import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter

from src import utils
from src.dataloader import load_data
from src.engine import evaluate
from src.train_one_epoch import train_one_epoch
from src.tSNE import TSNEVisualizer
from src.config_manager import ConfigManager
from src import factory
from src import checkpoint as checkpoint_utils


def main(args):
    writer = (
        SummaryWriter(
            f"./runs/{'_'.join(args.output_dir.split('/')[1:])}_{datetime.datetime.now().strftime('%b%d_%H%M')}"
        )
        if args.output_dir
        else None
    )

    if args.output_dir and not args.test_only:
        utils.mkdir(os.path.join(args.output_dir, "checkpoints"))

    utils.init_distributed_mode(args)
    from rich.table import Table
    from rich.console import Console
    from rich import box

    console = Console()
    console.clear()
    table = Table(
        title="Configuration",
        show_header=True,
        header_style="bold cyan",
        box=box.MINIMAL,
    )
    table.add_column("Parameter", style="cyan", width=25)
    table.add_column("Value", style="green")

    for arg, value in sorted(vars(args).items()):
        table.add_row(arg, str(value))

    console.print(table)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    (
        dataset,
        dataset_avg,
        dataset_val,
        dataset_test,
        train_sampler,
        train_avg_sampler,
        val_sampler,
        test_sampler,
    ) = load_data(
        *[os.path.join(args.data_path, x) for x in ["train", "val", "test"]], args
    )

    args.num_classes = len(dataset.classes)
    collate_fn = default_collate

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_avg = torch.utils.data.DataLoader(
        dataset_avg,
        batch_size=args.batch_size,
        sampler=train_avg_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = factory.create_model(args)
    criterion = factory.create_loss(args, model)
    optimizer = factory.create_optimizer(args, model, criterion)
    lr_scheduler = factory.create_scheduler(args, optimizer)

    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
        if len([x for x in criterion.parameters()]) > 0:
            criterion = torch.nn.parallel.DistributedDataParallel(criterion)
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(
            model_without_ddp, device=device, decay=1.0 - alpha
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.tsne:
        tSNE = TSNEVisualizer(data_loader_val, model_without_ddp, args)
        tSNE.visualize_with_tsne()
        return print("t-SNE visualization is saved.")

    if args.test_only:
        print("Start testing")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            meter = evaluate(
                model_ema,
                args.shot,
                args.val_iter,
                args,
                data_loader_avg,
                data_loader_val,
                device=device,
                header="Val (EMA)",
            )
            meter = evaluate(
                model_ema,
                args.shot,
                args.test_iter,
                args,
                data_loader_avg,
                data_loader_test,
                device=device,
                header="Test (EMA)",
            )
        else:
            meter = evaluate(
                model,
                args.shot,
                args.val_iter,
                args,
                data_loader_avg,
                data_loader_val,
                device=device,
                header="Val",
            )
            meter = evaluate(
                model,
                args.shot,
                args.test_iter,
                args,
                data_loader_avg,
                data_loader_test,
                device=device,
                header="Test",
            )
        console.print("[bold green]✓[/bold green] Testing completed!")
        return

    print("Start training")
    metric_logger = utils.MetricLogger(delimiter="  ")
    for s in args.shot:
        metric_logger.add_meter(f"best_shot{s}", utils.BestValue())
        if args.model_ema:
            metric_logger.add_meter(f"best_shot{s}_ema", utils.BestValue())
    start_time = time.time()

    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeRemainingColumn,
        TimeElapsedColumn,
    )
    from rich.console import Console
    from rich.live import Live
    from rich.console import Group
    from rich.table import Table

    class MutableRenderable:
        def __init__(self, renderable=None):
            self.renderable = renderable

        def __rich__(self):
            return self.renderable

        def update(self, renderable):
            self.renderable = renderable

    metrics_proxy = MutableRenderable(
        Table(title="Waiting for metrics...", show_header=False)
    )

    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="bold cyan"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.fields[info]}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    live_group = Group(progress, metrics_proxy)

    with Live(live_group, refresh_per_second=10, console=console) as live:
        epoch_task = progress.add_task(
            "Overall Progress",
            total=args.epochs - args.start_epoch,
            info=f"Epoch {0}/{args.epochs - args.start_epoch}",
        )

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            progress.update(
                epoch_task,
                info=f"Epoch {epoch - args.start_epoch + 1}/{args.epochs - args.start_epoch}",
            )

            meter = train_one_epoch(
                model,
                criterion,
                optimizer,
                data_loader,
                device,
                epoch,
                args,
                model_ema,
                scaler,
                writer,
                progress=progress,
                log_layout=metrics_proxy,
                epoch_task_id=epoch_task,
            )

            lr_scheduler.step()

            if (epoch + 1) % args.val_freq == 0:
                meter = evaluate(
                    model,
                    args.shot,
                    args.val_iter,
                    args,
                    data_loader_avg,
                    data_loader_val,
                    writer=writer,
                    epoch=epoch,
                    device=device,
                    header="Val",
                    progress=progress,
                    log_layout=metrics_proxy,
                )

                # Track best accuracy and save best checkpoint
                for s in args.shot:
                    current_acc = meter.meters[f"shot{s}_acc"].global_avg
                    metric_logger.meters[f"best_shot{s}"].update(current_acc)

                    if metric_logger.meters[f"best_shot{s}"].is_best:
                        checkpoint_utils.save_best_checkpoint(
                            model_without_ddp=model_without_ddp,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            epoch=epoch,
                            args=args,
                            output_dir=args.output_dir,
                            shot=s,
                            model_ema=model_ema,
                            scaler=scaler,
                            is_ema=False,
                        )

                if model_ema:
                    meter_ema = evaluate(
                        model_ema,
                        args.shot,
                        args.val_iter,
                        args,
                        data_loader_avg,
                        data_loader_val,
                        writer=writer,
                        epoch=epoch,
                        device=device,
                        header="Val (EMA)",
                        progress=progress,
                        log_layout=metrics_proxy,
                    )

                    # Track best EMA accuracy and save best EMA checkpoint
                    for s in args.shot:
                        current_acc_ema = meter_ema.meters[f"shot{s}_acc"].global_avg
                        metric_logger.meters[f"best_shot{s}_ema"].update(
                            current_acc_ema
                        )

                        # Check if this is the best EMA model so far
                        if metric_logger.meters[f"best_shot{s}_ema"].is_best:
                            checkpoint_utils.save_best_checkpoint(
                                model_without_ddp=model_without_ddp,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                epoch=epoch,
                                args=args,
                                output_dir=args.output_dir,
                                shot=s,
                                model_ema=model_ema,
                                scaler=scaler,
                                is_ema=True,
                            )

            progress.update(epoch_task, completed=epoch - args.start_epoch + 1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # Save final checkpoint
    if args.output_dir:
        checkpoint_utils.save_final_checkpoint(
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=args.epochs - 1,
            args=args,
            output_dir=args.output_dir,
            model_ema=model_ema,
            scaler=scaler,
        )

    console.print(
        f"[bold green]✓[/bold green] Training completed! Total time: {total_time_str}"
    )


if __name__ == "__main__":
    cm = ConfigManager()
    args = cm.get_args()
    main(args)
