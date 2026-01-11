import collections
import torch
from . import utils
from .metrics import meta_evaluate
from rich import box


def evaluate(
    model,
    shots,
    num_iter,
    eval_params,
    data_loader_avg,
    data_loader,
    device,
    epoch=0,
    print_freq=100,
    header="Val",
    writer=None,
    loss=None,
    progress=None,
    log_layout=None,
):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    with torch.inference_mode():
        train_avg = 0
        num = 0

        msg = f"{header} (stats)"
        if progress and hasattr(progress, "console"):
            progress.console.print(msg)
        else:
            print(msg)

        for image, _ in data_loader_avg:
            image = image.to(device, non_blocking=True)
            train_avg = train_avg + model(image)[0].mean(0)
            num += 1
        train_avg = utils.reduce_across_processes(train_avg / num, op="MEAN")

        test_embeddings, test_labels = [], []
        for image, target in metric_logger.log_every(
            data_loader, print_freq, header, progress, log_layout
        ):
            image, target = (
                image.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
            )
            output = model(image)[0]
            test_embeddings.append(output)
            test_labels.append(target)

        test_embeddings = utils.gather_across_processes(torch.cat(test_embeddings))
        test_labels = utils.gather_across_processes(torch.cat(test_labels))

    if utils.is_main_process():
        train_avg = train_avg.cpu().data.numpy()
        out_dict = collections.defaultdict(list)
        for out, label in zip(test_embeddings, test_labels):
            out_dict[label.item()].append(out.cpu().data.numpy())

        for s in shots:
            shot_info = meta_evaluate(out_dict, train_avg, s, num_iter, eval_params)
            metric_logger.meters[f"shot{s}_acc"].update(shot_info[0] * 100, n=1)
            metric_logger.meters[f"shot{s}_conf"].update(shot_info[1] * 100, n=1)
            if writer:
                writer.add_scalar(
                    f"{header}/Acc/shot_{s}",
                    metric_logger.meters[f"shot{s}_acc"].global_avg,
                    epoch,
                )
    else:
        for s in shots:
            metric_logger.meters[f"shot{s}_acc"].update(0, n=1)

    console = progress.console if progress and hasattr(progress, "console") else None

    if console:
        from rich.table import Table

        table = Table(
            title=f"{header} Results",
            show_header=True,
            header_style="bold cyan",
            box=box.HORIZONTALS,
        )
        table.add_column("Metric", style="dim", width=12)
        table.add_column("Value", justify="right", style="bold green")

        for k in sorted(metric_logger.meters.keys()):
            table.add_row(k, f"{metric_logger.meters[k].global_avg:.2f}")

        console.print(table)
        console.print()
    else:
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(
            title=f"{header} Results",
            show_header=True,
            header_style="bold cyan",
            box=box.HORIZONTALS,
        )
        table.add_column("Metric", style="dim", width=12)
        table.add_column("Value", justify="right", style="bold green")

        for k in sorted(metric_logger.meters.keys()):
            table.add_row(k, f"{metric_logger.meters[k].global_avg:.2f}")

        console.print(table)
        console.print()

    return metric_logger
