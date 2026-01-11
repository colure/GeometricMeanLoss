"""
This code is based on
https://github.com/pytorch/vision/tree/main/references/classification

modified by Takumi Kobayashi
Refactored by Tong Wu
"""

import copy
import datetime
import errno
import hashlib
import os
import time
from collections import OrderedDict, defaultdict, deque
from functools import reduce
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.nn
from torch.utils.tensorboard import SummaryWriter


class TensorWriter:
    def __init__(self, log_dir):
        if is_main_process():
            self.writer = SummaryWriter(log_dir=log_dir)

    def update(self, meter, epoch, suffix="train"):
        if is_main_process():
            for k in meter.meters.keys():
                self.writer.add_scalar(
                    k + "/" + suffix, meter.meters[k].global_avg, epoch
                )
            self.writer.close()


class BestValue:
    def __init__(self):
        self.best_value = -1
        self.is_best = False

    def update(self, val):
        self.is_best = val > self.best_value
        self.best_value = max(self.best_value, val)


class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(
        self, iterable, print_freq, header=None, progress=None, log_layout=None
    ):
        from rich.progress import (
            Progress,
            SpinnerColumn,
            BarColumn,
            TextColumn,
            TimeRemainingColumn,
            TimeElapsedColumn,
        )
        from rich.console import Console
        from rich.table import Table
        from rich import box

        i = 0
        header = header or ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        MB = 1024.0 * 1024.0

        created_progress = False
        if progress is None:
            created_progress = True
            console = Console()
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(complete_style="green", finished_style="bold green"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                TextColumn("{task.fields[info]}"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
                transient=False,
            )
            progress.start()

        try:
            task = progress.add_task(header, total=len(iterable), info="")
            for obj in iterable:
                data_time.update(time.time() - end)
                yield obj
                iter_time.update(time.time() - end)

                if i % print_freq == 0:
                    if log_layout is not None:
                        table = Table(
                            show_header=True,
                            header_style="bold magenta",
                            box=box.MINIMAL,
                        )
                        table.add_column("Metric", style="cyan", no_wrap=True)
                        for name in self.meters.keys():
                            table.add_column(name, justify="right")
                        table.add_column("Time", justify="right")
                        table.add_column("Data", justify="right")
                        if torch.cuda.is_available():
                            table.add_column("Mem", justify="right")

                        row_data = ["Value"]
                        for meter in self.meters.values():
                            row_data.append(f"{meter.value:.4f}")
                        row_data.append(f"{iter_time.avg:.4f}s")
                        row_data.append(f"{data_time.avg:.4f}s")
                        if torch.cuda.is_available():
                            row_data.append(
                                f"{torch.cuda.max_memory_allocated() / MB:.0f}MB"
                            )

                        table.add_row(*row_data)
                        log_layout.update(table)
                        progress.update(task, completed=i, info="")
                    else:
                        metrics_parts = [
                            f"{name}: {meter.value:.4f}"
                            for name, meter in self.meters.items()
                        ]
                        metrics_parts.extend(
                            [
                                f"time: {iter_time.avg:.4f}s",
                                f"data: {data_time.avg:.4f}s",
                            ]
                        )
                        if torch.cuda.is_available():
                            metrics_parts.append(
                                f"mem: {torch.cuda.max_memory_allocated() / MB:.0f}MB"
                            )
                        progress.update(
                            task, completed=i, info=" | ".join(metrics_parts)
                        )

                i += 1
                end = time.time()
            progress.update(task, completed=len(iterable))
        finally:
            if created_progress:
                progress.stop()
            else:
                progress.remove_task(task)

        total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        if hasattr(progress, "console"):
            progress.console.print(
                f"[bold green]✓[/bold green] {header} Total time: {total_time_str}\n"
            )
        else:
            print(f"{header} Total time: {total_time_str}\n")


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def accuracy(output, target, topk=(1,)):
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def average_checkpoints(inputs):
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        if new_state is None:
            new_state = state
        model_params = state["model"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(f"Mismatch in params: {params_keys} vs {model_params_keys}")
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
            else:
                params_dict[k] += p
    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def store_model_weights(model, checkpoint_path, checkpoint_key="model", strict=True):
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = os.path.dirname(checkpoint_path)
    model = copy.deepcopy(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint_key == "model_ema":
        del checkpoint[checkpoint_key]["n_averaged"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            checkpoint[checkpoint_key], "module."
        )
    model.load_state_dict(checkpoint[checkpoint_key], strict=strict)
    tmp_path = os.path.join(output_dir, str(model.__hash__()))
    torch.save(model.state_dict(), tmp_path)
    sha256_hash = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hh = sha256_hash.hexdigest()
    output_path = os.path.join(output_dir, "weights-" + str(hh[:8]) + ".pth")
    os.replace(tmp_path, output_path)
    return output_path


def reduce_across_processes(val, op="SUM"):
    if not is_dist_avail_and_initialized():
        return val if type(val) is torch.Tensor else torch.tensor(val)
    t = val if type(val) is torch.Tensor else torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t if op == "SUM" else t / get_world_size()


def gather_across_processes(x):
    if not is_dist_avail_and_initialized():
        return x
    return torch.cat(torch.distributed.nn.functional.all_gather(x))


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)
    params = {"other": [], "norm": []}
    params_weight_decay = {"other": weight_decay, "norm": norm_weight_decay}
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = (
                    f"{prefix}.{name}" if prefix != "" and "." in key else name
                )
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)
        for child_name, child_module in module.named_children():
            _add_params(
                child_module,
                prefix=f"{prefix}.{child_name}" if prefix != "" else child_name,
            )

    _add_params(model)
    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append(
                {"params": params[key], "weight_decay": params_weight_decay[key]}
            )
    return param_groups


def load_state_dict_finetune(net, new_state_dict):
    old_state_dict = net.state_dict()
    inconsist_layer = []
    for k in new_state_dict.keys():
        if new_state_dict[k].shape != old_state_dict[k].shape:
            l = ".".join(k.split(sep=".")[:-1])
            new_state_dict[k] = old_state_dict[k].clone()
            inconsist_layer.append(l)
    net.load_state_dict(new_state_dict)
    return net, inconsist_layer[-1]


def ema_cleanup(model):
    return {k[7:]: v for k, v in model.items() if k.startswith("module.")}


def set_module_by_name(module, access_string, new_layer):
    names = access_string.split(sep=".")
    parent = reduce(getattr, names[:-1], module)
    old_layer = getattr(parent, names[-1])
    setattr(parent, names[-1], new_layer)
    return old_layer
