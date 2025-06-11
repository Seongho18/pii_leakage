import os

from ..arguments.env_args import EnvArgs
from ..arguments.model_args import ModelArgs
from ..arguments.sampling_args import SamplingArgs
from .dataset import Dataset


class GeneratedDataset(Dataset):

    def __init__(self, model_args: ModelArgs, sampling_args: SamplingArgs, env_args: EnvArgs = None):
        """ A generated dataset is identified by the dataset args and sampling args. """
        super().__init__(env_args=env_args)
        self.sampling_args = sampling_args
        self.model_args = model_args

    def _load_base_dataset(self):
        raise NotImplementedError


