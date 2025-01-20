import numpy as np
from pathlib import Path
import torch
from torch.utils.data import IterableDataset


class BlockStreamDataset(IterableDataset):

    def __init__(self, path: Path, block_size: int):
        super().__init__()
        self.path = path
        self.block_size = block_size

    def generate(self):
        while True:
            data = np.memmap(self.path, dtype=np.uint16, mode="r")
            ix = torch.randint(len(data) - self.block_size, (1,))
            block = torch.from_numpy(
                (data[ix:ix + self.block_size].astype(np.int64))
            )
            yield block

    def __iter__(self):
        return iter(self.generate())
