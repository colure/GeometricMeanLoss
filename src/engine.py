import collections
import torch
from . import utils
from .metrics import meta_evaluate
from rich import box
from rich.console import Console
from rich.table import Table


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
        train_avg = None
        num = 0

        msg = f"{header} (stats)"
        if progress and hasattr(progress, "console"):
            progress.console.print(msg)
        else:
            print(msg)

        for image, _ in data_loader_avg:
            image = image.to(device, non_blocking=True)
            batch_avg = model(image)[0].mean(0)
            train_avg = batch_avg if train_avg is None else train_avg + batch_avg
            num += 1
        train_avg = utils.reduce_across_processes(train_avg / num, op="MEAN")

        embedding_buffer = None
        label_buffer = None
        offset = 0
        for image, target in metric_logger.log_every(
            data_loader, print_freq, header, progress, log_layout
        ):
            image, target = (
                image.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
            )
            output = model(image)[0]
            batch_size = output.size(0)
            if embedding_buffer is None:
                local_sample_count = len(data_loader.sampler)
                embedding_buffer = output.new_empty(
                    (local_sample_count, output.size(-1))
                )
                label_buffer = target.new_empty(local_sample_count)

            next_offset = offset + batch_size
            embedding_buffer[offset:next_offset].copy_(output)
            label_buffer[offset:next_offset].copy_(target)
            offset = next_offset

        test_embeddings = utils.gather_across_processes(embedding_buffer[:offset])
        test_labels = utils.gather_across_processes(label_buffer[:offset])

    if utils.is_main_process():
        train_avg = train_avg.cpu().numpy()
        test_embeddings_np = test_embeddings.cpu().numpy()
        test_labels_np = test_labels.cpu().numpy()
        out_dict = collections.defaultdict(list)
        for out, label in zip(test_embeddings_np, test_labels_np, strict=True):
            out_dict[int(label)].append(out)

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
            metric_logger.meters[f"shot{s}_conf"].update(0, n=1)

    console = (
        progress.console if progress and hasattr(progress, "console") else Console()
    )

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
