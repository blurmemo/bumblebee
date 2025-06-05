from collections.abc import Iterator
from typing import Optional

import torch


class DistributedSampler(torch.utils.data.DistributedSampler):
    def __init__(
        self,
        dataset: torch.utils.data.dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

    def __iter__(self) -> Iterator[torch.utils.data.distributed._T_co]:
        indices = super().__iter__()

        ##### add #####
        # Every epoch execute this function once.
        # In order not to call `set_epoch` manually every epoch, it will auto add by one.
        self.set_epoch(self.epoch+1)
        ##### add #####

        return indices




class DataloaderRegistry:
    def __init__(self, dataloader: torch.utils.data.DataLoader = None, batch_size: int = 1, shuffle: bool = True,):
        self.dataloader = dataloader
        self.shuffle = shuffle
        self.batch_size = batch_size


    def validate_init_params(self):
        if self.dataloader is None:
            raise ValueError("`dataloader` must be required but None.")

        if self.batch_size < 1:
            raise ValueError(f"`batch_size`: {self.batch_size} must be >= 1.")


    @property
    def sampler(self):
        return self.dataloader.sampler

    def __call__(self, *args, **kwargs):
        dataset = self.dataloader.dataset
        num_workers = self.dataloader.num_workers
        pin_memory = self.dataloader.pin_memory
        batch_size = self.batch_size
        drop_last = self.dataloader.drop_last
        collate_fn = self.dataloader.collate_fn


        return torch.utils.data.DataLoader(
            dataset,
            batch_size,
            sampler=self.sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )


class DistributedDataLoaderRegistry(DataloaderRegistry):
    """
    Note:
        shuffle: bool True means shuffle samplers as `dataloader` shuffle, False means no shuffle even though `dataloader` shuffle
        drop_last: It will do the same `drop_last` operation two times in `dataloader` and `DistributedDataLoader`
    """
    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader = None,
        batch_size: int = 1,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
    ):
        super().__init__(dataloader, batch_size, shuffle)
        self.rank = rank
        self.world_size = world_size


    def validate_init_params(self):
        super().validate_init_params()
        
        if self.world_size <= 1:
            raise ValueError(f"`world_size`: {self.world_size} must be greater than 1 for `DistributedDataLoader`.")

    @property
    def sampler(self):
        dist_sampler = DistributedSampler(
            self.dataloader.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=self.shuffle,
            drop_last=self.dataloader.drop_last
        )
        return dist_sampler

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)