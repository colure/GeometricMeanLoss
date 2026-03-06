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
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(len(self.dataset))

        indices = indices.repeat_interleave(self.repetitions)
        padding_size = self.total_size - indices.numel()
        if padding_size > 0:
            indices = torch.cat((indices, indices[:padding_size]))
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices[: self.num_selected_samples].tolist())

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class ClassAwareDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, class_per_batch, sample_per_class, **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        self.y = torch.tensor([y[1] for y in dataset.samples])
        self.classes = self.y.unique()
        self.class_indices = [torch.nonzero(self.y == c).flatten() for c in self.classes]
        max_samp_num = max(len(indices) for indices in self.class_indices)
        num_samples = max_samp_num * len(self.class_indices)
        self.num_samples = math.ceil(num_samples / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.class_per_batch = class_per_batch
        self.sample_per_class = sample_per_class

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
        batch_size_per_replica = (
            self.class_per_batch * self.sample_per_class // self.num_replicas
        )
        return iter(self.get_sublist(indices, batch_size_per_replica))

    def class_aware_shuffle(self, g):
        max_samp_num = max(len(indices) for indices in self.class_indices)
        shuffled_by_class = torch.stack(
            [self.randshuffle(self.append(indices, max_samp_num, g), g) for indices in self.class_indices]
        )
        grouped_samples = [
            self.randshuffle(chunk, g).flatten()
            for chunk in self.split(shuffled_by_class, self.sample_per_class)
        ]
        flat_indices = torch.cat(grouped_samples)
        return torch.cat(
            [
                self.randshuffle(chunk, g)
                for chunk in self.split(
                    flat_indices, self.class_per_batch * self.sample_per_class
                )
            ]
        )

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
