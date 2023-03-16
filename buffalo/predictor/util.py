"""
This module contains helper functions to manipulate predictors.
"""

import numpy as np

from typing import Optional, Union
from torch.utils.data import Dataset, DataLoader, Sampler

class TimeSeries:

    class TimeSeriesDataset(Dataset):
        def __init__(self, data, window_size):
            self.data = data
            self.window_size = window_size

        def __len__(self):
            return len(self.data) - self.window_size

        def __getitem__(self, index):
            start_index = index
            end_index = index + self.window_size
            x = self.data[start_index:end_index]
            y = self.data[end_index]
            return x, y
    
    class TimeSeriesDataLoad(DataLoader):
        def __init__(self, dataset, batch_size, sampler, batch_sampler, num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None, pin_memory: bool = False, drop_last: bool = False, timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None, multiprocessing_context=None, generator=None, *, prefetch_factor: int = 2, persistent_workers: bool = False, pin_memory_device: str = ""):
            super().__init__(dataset, batch_size, False, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, prefetch_factor=prefetch_factor, persistent_workers, pin_memory_device)
    
    class TimeSeriesSampler(Sampler):

        def __init__(self, data_source) -> None:
            super().__init__(data_source)

    
    def __init__(self, endog, exog, batch_size, pin_memory, device) -> None:
        pass