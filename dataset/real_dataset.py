import os
from copy import deepcopy

from datasets import load_dataset

from ..arguments.dataset_args import DatasetArgs
from ..arguments.env_args import EnvArgs
from .dataset import Dataset


class RealDataset(Dataset):
    """ A lightweight wrapper around a huggingface text dataset
    that allows caching and indexing PII
    """

    def __init__(self, dataset_args: DatasetArgs, env_args: EnvArgs = None):
        """ A real dataset is a dataset loaded from real training data.
        """
        self.dataset_args = dataset_args
        super().__init__(env_args)
        self._tokenized_base_dataset = self._load_base_dataset()  # this is the reference to the huggingface dataset

    def _load_base_dataset(self, force_undefended=False):
        """ Loads the huggingface dataset. """
        print(self.dataset_args.dataset_path)

        return load_dataset(self.dataset_args.dataset_path, cache_dir=self.dataset_args.cache_dir(),
                            name=self.dataset_args.dataset_mode if not force_undefended else "undefended",
                            sample_duplication_rate=self.dataset_args.sample_duplication_rate,
                            trust_remote_code=True)[self.dataset_args.split]

    def shuffle(self):
        self._base_dataset.shuffle()
        return self

    def copy(self):
        return deepcopy(self)

    def select(self, indices):
        clone = self.copy()
        clone._tokenized_base_dataset = clone._tokenized_base_dataset.select(indices)
        return clone

    def __iter__(self):
        return self._base_dataset.__iter__()

    def __getitem__(self, item):
        return self._base_dataset.__getitem__(item)

    def get_hf_dataset(self):
        return self._base_dataset

    def first(self, column_name="text"):
        return self[column_name][0]

