import torch
import loss
import models
from src import utils


def create_model(args):
    print()
    print("Initializing model...\n")
    model = getattr(models, args.model)(
        num_classes=args.num_classes,
        feature_dim=args.projection_feat_dim,
        projection=args.projection,
        use_fc=False,
    )
    device = torch.device(args.device)
    model.to(device)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model


def create_loss(args, model=None):
    class_proxy_ = (
        [args.num_classes, args.projection_feat_dim] if args.class_proxy else None
    )

    criterion = loss.WrapperLoss(
        loss=getattr(loss, args.loss)(**vars(args)), class_proxy=class_proxy_
    )

    device = torch.device(args.device)
    criterion.to(device)

    return criterion


def create_optimizer(args, model, criterion):
    parameters = utils.set_weight_decay(model, args.weight_decay)
    parameters = parameters + utils.set_weight_decay(criterion, args.weight_decay)

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            eps=0.0316,
            alpha=0.9,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            parameters, lr=args.lr, weight_decay=args.weight_decay
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        raise RuntimeError(
            f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported."
        )

    return optimizer


def create_scheduler(args, optimizer):
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_gamma
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[args.lr_warmup_epochs],
        )
    else:
        lr_scheduler = main_lr_scheduler

    return lr_scheduler
