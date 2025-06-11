from abc import abstractmethod

from ..arguments.env_args import EnvArgs


class Dataset:
    def __init__(self, env_args: EnvArgs):
        self.env_args = env_args if env_args is not None else EnvArgs()

        self._base_dataset = self._load_base_dataset()   # The dataset that this class wraps around (e.g., huggingface)

    @abstractmethod
    def _load_base_dataset(self):
        """ Loads the underlying dataset. """
        raise NotImplementedError

    def __len__(self):
        return len(self._base_dataset)

    @property
    def _pii_cache(self):
        raise NotImplementedError

