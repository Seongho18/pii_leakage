import os

import datasets
import pandas as pd

from pii_leakage.extern.CustomBuilder import CustomBuilder
from pii_leakage.utils.random import rnd_idx


class CustomEnron(datasets.GeneratorBasedBuilder):
    """ A wrapper around the Enron e-mail dataset that uses anonymization.  """

    VERSION = datasets.Version("1.0.0")
    _DESCRIPTION = "A custom wrapper for the Enron dataset."
    _TEXT = "text"

    _PATH = {
        "path": "../dataset/enron"
    }

    BUILDER_CONFIGS = [
        CustomBuilder(name="undefended", sample_duplication_rate=1, version=VERSION,
                          description="undefended, private data"),
        CustomBuilder(name="scrubbed", sample_duplication_rate=1, version=VERSION,
                          description="PII replaced with anon token")
    ]
    DEFAULT_CONFIG_NAME = "undefended"

    def __init__(self, *args, **kwargs):
        self.df: pd.DataFrame = pd.DataFrame()
        super().__init__(*args, **kwargs)

    def _info(self):
        features = datasets.Features({self._TEXT: datasets.Value("string")})
        return datasets.DatasetInfo(
            description=self._DESCRIPTION,
            features=features
        )

    def _travel_dirs(self, path):
        file_list = []
        if os.path.isdir(path):
            for name in os.listdir(path):
                inner_path = os.path.join(path, name)
                file_list.extend(self._travel_dirs(inner_path))
        else:
            file_list.append(path)
        return file_list

    def _split_generators(self, dl_manager):
        path = self._PATH["path"]
        self.data = []
        file_list = self._travel_dirs(path)
        for file_path in sorted(file_list):
            with open(file_path, 'r', encoding='iso-8859-1') as f:
                try:
                    lines = f.readlines()
                except UnicodeDecodeError:
                    print(f"UnicodeDecodeError :\t{file_path}")
                    continue
            header_ends = False
            for index_l, line in enumerate(lines):
                if line[:11] == "X-FileName:":
                    header_ends = True
                    continue
                if header_ends:
                    if line.strip() == "":
                        continue
                    break
            text = "".join(lines[index_l:])
            self.data.append(text)

        if self.config.shuffle_facts_seed > 0:
            self.data = [self.data[i] for i in rnd_idx(N=len(self.data), seed=self.config.shuffle_facts_seed)]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "start": 0.0,
                    "end": 0.45
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "start": 0.45,
                    "end": 0.55
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "validation",
                    "start": 0.55,
                    "end": 1.0
                },
            ),
        ]
