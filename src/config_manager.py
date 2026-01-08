import argparse
import os
import tomli

DEFAULTS = {
    "data_path": "data/mini_imagenet",
    "val_resize_size": 84,
    "val_crop_size": 84,
    "train_crop_size": 84,
    "model": "resnet12",
    "projection": True,
    "projection_feat_dim": 192,
    "model_ema_steps": 32,
    "model_ema_decay": 0.99998,
    "amp": True,
    "device": "cuda",
    "workers": 16,
    "batch_size": 128,
    "epochs": 120,
    "opt": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "lr_scheduler": "steplr",
    "lr_step_size": 84,
    "lr_gamma": 0.01,
    "lr_min": 0.0,
    "lr_warmup_epochs": 10,
    "lr_warmup_method": "linear",
    "lr_warmup_decay": 0.033,
    "clip_grad_norm": None,
    "backend": "PIL",
    "class_aware_sampler": None,
    "output_dir": None,
    "resume": "",
    "print_freq": 10,
    "val_freq": 5,
    "start_epoch": 0,
    "auto_augment": None,
    "ra_magnitude": 9,
    "augmix_severity": 3,
    "random_erase": 0.0,
    "ra_reps": 3,
    "interpolation": "bilinear",
    "world_size": 1,
    "dist_url": "env://",
    "loss": "GMLoss",
    "logit": "l1_dist",
    "logit_temperature": 1.0,
    "classifier": "nc",
    "num_NN": 1,
    "test_iter": 10000,
    "val_iter": 3000,
    "test_way": 5,
    "test_query": 15,
    "shot": "1,5",
    "eval_norm_type": "CCOS",
    "norm": 2.0,
}


class ConfigManager:
    def __init__(self, config_file="config.toml"):
        self.config_file = config_file
        self.config = self._load_config()
        self.parser = self._get_parser()
        self.args = self.parser.parse_args()

        # Post-processing
        if isinstance(self.args.shot, str):
            self.args.shot = [int(x) for x in self.args.shot.split(",")]

    def _load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "rb") as f:
                return tomli.load(f)
        return {}

    def _get_val(self, section, key, default_key):
        """Retrieve value from config -> DEFAULTS -> None"""
        val = self.config.get(section, {}).get(key)
        if val is None:
            return DEFAULTS.get(default_key)
        return val

    def _get_parser(self):
        parser = argparse.ArgumentParser(description="PyTorch Few-Shot Training")

        # Data
        parser.add_argument(
            "--data-path",
            default=self._get_val("data", "path", "data_path"),
            type=str,
            help="dataset path",
        )
        parser.add_argument(
            "--val-resize-size",
            default=self._get_val("data", "val_resize_size", "val_resize_size"),
            type=int,
        )
        parser.add_argument(
            "--val-crop-size",
            default=self._get_val("data", "val_crop_size", "val_crop_size"),
            type=int,
        )
        parser.add_argument(
            "--train-crop-size",
            default=self._get_val("data", "train_crop_size", "train_crop_size"),
            type=int,
        )
        parser.add_argument(
            "--backend",
            default=self._get_val("data", "backend", "backend"),
            type=str.lower,
        )
        parser.add_argument(
            "--class-aware-sampler",
            default=self._get_val("data", "class_aware_sampler", "class_aware_sampler"),
            type=str,
        )
        parser.add_argument(
            "--auto-augment",
            default=self._get_val("data", "auto_augment", "auto_augment"),
            type=str,
        )
        parser.add_argument(
            "--ra-magnitude",
            default=self._get_val("data", "ra_magnitude", "ra_magnitude"),
            type=int,
        )
        parser.add_argument(
            "--augmix-severity",
            default=self._get_val("data", "augmix_severity", "augmix_severity"),
            type=int,
        )
        parser.add_argument(
            "--random-erase",
            default=self._get_val("data", "random_erase", "random_erase"),
            type=float,
        )
        parser.add_argument(
            "--ra-sampler",
            action="store_true",
            default=self._get_val("data", "ra_sampler", None),
        )  # boolean in toml
        parser.add_argument(
            "--ra-reps", default=self._get_val("data", "ra_reps", "ra_reps"), type=int
        )
        parser.add_argument(
            "--interpolation",
            default=self._get_val("data", "interpolation", "interpolation"),
            type=str,
        )

        # Model
        parser.add_argument(
            "--model", default=self._get_val("model", "name", "model"), type=str
        )
        parser.add_argument(
            "--projection",
            default=self._get_val("model", "projection", "projection"),
            type=bool,
        )  # Argparse bool is tricky, but here it's default value
        parser.add_argument(
            "--projection-feat-dim",
            default=self._get_val(
                "model", "projection_feat_dim", "projection_feat_dim"
            ),
            type=int,
        )
        parser.add_argument(
            "--model-ema",
            action="store_true",
            default=self._get_val("model", "ema", None),
        )
        parser.add_argument(
            "--model-ema-steps",
            default=self._get_val("model", "ema_steps", "model_ema_steps"),
            type=int,
        )
        parser.add_argument(
            "--model-ema-decay",
            default=self._get_val("model", "ema_decay", "model_ema_decay"),
            type=float,
        )

        # Training
        parser.add_argument(
            "--amp", default=self._get_val("training", "amp", "amp"), type=bool
        )
        parser.add_argument(
            "--device", default=self._get_val("training", "device", "device"), type=str
        )
        parser.add_argument(
            "-j",
            "--workers",
            default=self._get_val("training", "workers", "workers"),
            type=int,
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            default=self._get_val("training", "batch_size", "batch_size"),
            type=int,
        )
        parser.add_argument(
            "--epochs", default=self._get_val("training", "epochs", "epochs"), type=int
        )
        parser.add_argument(
            "--opt", default=self._get_val("training", "optimizer", "opt"), type=str
        )
        parser.add_argument(
            "--lr", default=self._get_val("training", "lr", "lr"), type=float
        )
        parser.add_argument(
            "--momentum",
            default=self._get_val("training", "momentum", "momentum"),
            type=float,
        )
        parser.add_argument(
            "--wd",
            "--weight-decay",
            default=self._get_val("training", "weight_decay", "weight_decay"),
            type=float,
            dest="weight_decay",
        )
        parser.add_argument(
            "--lr-scheduler",
            default=self._get_val("training", "lr_scheduler", "lr_scheduler"),
            type=str,
        )
        parser.add_argument(
            "--lr-step-size",
            default=self._get_val("training", "lr_step_size", "lr_step_size"),
            type=int,
        )
        parser.add_argument(
            "--lr-gamma",
            default=self._get_val("training", "lr_gamma", "lr_gamma"),
            type=float,
        )
        parser.add_argument(
            "--lr-min",
            default=self._get_val("training", "lr_min", "lr_min"),
            type=float,
        )
        parser.add_argument(
            "--lr-warmup-epochs",
            default=self._get_val("training", "lr_warmup_epochs", "lr_warmup_epochs"),
            type=int,
        )
        parser.add_argument(
            "--lr-warmup-method",
            default=self._get_val("training", "lr_warmup_method", "lr_warmup_method"),
            type=str,
        )
        parser.add_argument(
            "--lr-warmup-decay",
            default=self._get_val("training", "lr_warmup_decay", "lr_warmup_decay"),
            type=float,
        )
        parser.add_argument(
            "--clip-grad-norm",
            default=self._get_val("training", "clip_grad_norm", "clip_grad_norm"),
            type=float,
        )
        parser.add_argument(
            "--loss", default=self._get_val("training", "loss", "loss"), type=str
        )
        parser.add_argument(
            "--logit", default=self._get_val("training", "logit", "logit"), type=str
        )
        parser.add_argument(
            "--logit-temperature",
            default=self._get_val("training", "logit_temperature", "logit_temperature"),
            type=float,
            dest="T",
        )
        parser.add_argument(
            "--class-proxy",
            action="store_true",
            default=self._get_val("training", "class_proxy", None),
        )
        parser.add_argument(
            "--norm", default=self._get_val("training", "norm", "norm"), type=float
        )

        # Distributed
        parser.add_argument(
            "--world-size",
            default=self._get_val("distributed", "world_size", "world_size"),
            type=int,
        )
        parser.add_argument(
            "--dist-url",
            default=self._get_val("distributed", "dist_url", "dist_url"),
            type=str,
        )

        # Logging
        parser.add_argument(
            "--output-dir",
            default=self._get_val("logging", "output_dir", "output_dir"),
            type=str,
        )
        parser.add_argument(
            "--resume", default=self._get_val("logging", "resume", "resume"), type=str
        )
        parser.add_argument(
            "--print-freq",
            default=self._get_val("logging", "print_freq", "print_freq"),
            type=int,
        )
        parser.add_argument(
            "--val-freq",
            default=self._get_val("logging", "val_freq", "val_freq"),
            type=int,
        )
        parser.add_argument(
            "--start-epoch",
            default=self._get_val("logging", "start_epoch", "start_epoch"),
            type=int,
        )
        parser.add_argument(
            "--save-all-checkpoints",
            action="store_true",
            default=self._get_val("logging", "save_all_checkpoints", None),
        )

        # Evaluation
        parser.add_argument(
            "--test-only",
            action="store_true",
            default=self._get_val("evaluation", "test_only", None),
        )
        parser.add_argument(
            "--classifier",
            default=self._get_val("evaluation", "classifier", "classifier"),
            type=str,
        )
        parser.add_argument(
            "--num-NN",
            default=self._get_val("evaluation", "num_NN", "num_NN"),
            type=int,
        )
        parser.add_argument(
            "--median-prototype",
            action="store_true",
            default=self._get_val("evaluation", "median_prototype", None),
        )
        parser.add_argument(
            "--test-iter",
            default=self._get_val("evaluation", "test_iter", "test_iter"),
            type=int,
        )
        parser.add_argument(
            "--val-iter",
            default=self._get_val("evaluation", "val_iter", "val_iter"),
            type=int,
        )
        parser.add_argument(
            "--test-way",
            default=self._get_val("evaluation", "test_way", "test_way"),
            type=int,
        )
        parser.add_argument(
            "--test-query",
            default=self._get_val("evaluation", "test_query", "test_query"),
            type=int,
        )
        parser.add_argument(
            "--shot", default=self._get_val("evaluation", "shot", "shot"), type=str
        )
        parser.add_argument(
            "--eval-norm-type",
            default=self._get_val("evaluation", "eval_norm_type", "eval_norm_type"),
            type=str,
        )
        parser.add_argument(
            "--tsne",
            action="store_true",
            default=self._get_val("evaluation", "tsne", None),
        )

        # Misc
        parser.add_argument(
            "--use-deterministic-algorithms",
            action="store_true",
            help="Forces the use of deterministic algorithms only.",
        )

        return parser

    def get_args(self):
        return self.args


if __name__ == "__main__":
    cm = ConfigManager()
    print(cm.get_args())
