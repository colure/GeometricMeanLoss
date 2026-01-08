import math
import torch
import torch.distributed as dist


class RASampler(torch.utils.data.Sampler):
    def __init__(
        self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() else 0
        self.dataset, self.num_replicas, self.rank = dataset, num_replicas, rank
        self.epoch, self.shuffle, self.seed, self.repetitions = (
            0,
            shuffle,
            seed,
            repetitions,
        )
        self.num_samples = int(
            math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(
            math.floor(len(self.dataset) // 256 * 256 / self.num_replicas)
        )

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        indices = [ele for ele in indices for _ in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class ClassAwareDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, class_per_batch, sample_per_class, **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        self.y = torch.tensor([y[1] for y in dataset.samples])
        max_samp_num = max([(self.y == c).sum().item() for c in self.y.unique()])
        num_samples = max_samp_num * len(self.y.unique())
        self.num_samples = (
            math.ceil(num_samples / self.num_replicas)
            if self.drop_last
            else math.ceil(num_samples / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas
        self.samples_in_batch = [class_per_batch, sample_per_class]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = self.class_aware_shuffle(g).tolist()
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[: self.total_size]
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(
            self.get_sublist(
                indices,
                self.samples_in_batch[0]
                * self.samples_in_batch[1]
                // self.num_replicas,
            )
        )

    def class_aware_shuffle(self, g):
        bc, bn = self.samples_in_batch
        cls_to_ind = [torch.nonzero(self.y == c).squeeze() for c in self.y.unique()]
        max_samp_num = max([len(x) for x in cls_to_ind])
        cls_to_ind = torch.stack(
            [self.randshuffle(self.append(x, max_samp_num, g), g) for x in cls_to_ind]
        )
        D = [self.randshuffle(x, g).flatten() for x in self.split(cls_to_ind, bn)]
        batches = torch.cat(
            [self.randshuffle(x, g) for x in self.split(torch.cat(D), bc * bn)]
        )
        return batches

    def randshuffle(self, x, g):
        return x[torch.randperm(len(x), generator=g)]

    def append(self, x, n, g):
        return torch.cat(
            [x.repeat(n // len(x)), self.randshuffle(x, g)[: (n % len(x))]]
        )

    def split(self, x, num, dim=-1):
        return torch.tensor_split(x, torch.arange(0, x.size(dim), num)[1:], dim)

    def get_sublist(self, lst, a):
        return lst[: len(lst) - (len(lst) % a)]
