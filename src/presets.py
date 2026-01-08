import torch
from torchvision.transforms.functional import InterpolationMode


def get_module(use_v2):
    if use_v2:
        import torchvision.transforms.v2 as T

        return T
    import torchvision.transforms as T

    return T


class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        backend="pil",
        use_v2=False,
    ):
        T = get_module(use_v2)
        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms.append(T.RandomResizedCrop(crop_size, interpolation=interpolation))
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))

        if auto_augment_policy:
            if auto_augment_policy == "ra":
                transforms.append(
                    T.RandAugment(interpolation=interpolation, magnitude=ra_magnitude)
                )
            elif auto_augment_policy == "ta_wide":
                transforms.append(T.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                transforms.append(
                    T.AugMix(interpolation=interpolation, severity=augmix_severity)
                )
            else:
                transforms.append(
                    T.AutoAugment(
                        policy=T.AutoAugmentPolicy(auto_augment_policy),
                        interpolation=interpolation,
                    )
                )

        if backend == "pil":
            transforms.append(T.PILToTensor())
        transforms.extend(
            [T.ConvertImageDtype(torch.float), T.Normalize(mean=mean, std=std)]
        )
        if random_erase_prob > 0:
            transforms.append(T.RandomErasing(p=random_erase_prob))
        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        backend="pil",
        use_v2=False,
    ):
        T = get_module(use_v2)
        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms.extend(
            [
                T.Resize(resize_size, interpolation=interpolation),
                T.CenterCrop(crop_size),
            ]
        )
        if backend == "pil":
            transforms.append(T.PILToTensor())
        transforms.extend(
            [T.ConvertImageDtype(torch.float), T.Normalize(mean=mean, std=std)]
        )
        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)
