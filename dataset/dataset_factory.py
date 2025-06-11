from ..arguments.dataset_args import DatasetArgs
from ..arguments.env_args import EnvArgs
from .real_dataset import RealDataset


class DatasetFactory:

    @staticmethod
    def from_dataset_args(dataset_args: DatasetArgs, env_args: EnvArgs = None) -> RealDataset:
        return RealDataset(dataset_args, env_args=env_args)


