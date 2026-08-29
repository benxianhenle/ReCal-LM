"""Dataset utilities for weighted sampling across multiple sources.

中文：用于在多个数据源之间按权重采样的数据集工具。"""

import random
from torch.utils.data import Dataset


class WeightedMixtureDataset(Dataset):
    """Virtual dataset that samples one child dataset per requested item.

中文：每次取样时按权重选择一个子数据集的虚拟数据集。"""

    def __init__(self, datasets: list[Dataset], weights: list[float], size: int):
        """Validate source weights and set the virtual epoch size.

中文：校验数据源权重，并设置虚拟 epoch 大小。"""

        if len(datasets) != len(weights):
            raise ValueError("datasets and weights must have the same length")
        self.datasets = datasets
        self.weights = weights
        self.size = size

    def __len__(self) -> int:
        """Return the configured virtual length of the mixture.

中文：返回混合数据集配置的虚拟长度。"""

        return self.size

    def __getitem__(self, index: int):
        """Draw a source dataset by weight and return a random item from it.

中文：按权重抽取一个源数据集，并从中随机返回一条样本。"""

        dataset = random.choices(self.datasets, weights=self.weights, k=1)[0]
        return dataset[random.randrange(len(dataset))]
