import os

import datasets
import pandas as pd

from pii_leakage.extern.CustomBuilder import CustomBuilder
from pii_leakage.utils.random import rnd_idx


class CustomTrec(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")
    _DESCRIPTION = "A custom wrapper for the Trec dataset."
    _TEXT = "text"

    _PATH = {
        "path": "../dataset/trec_deduplicated.txt"
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
        with open(path, 'r', encoding='iso-8859-1') as f:
            lines = f.readlines()
        text= ""
        for line in lines:
            if len(text) + len(line)> 5000:
                self.data.append(text)
                text = line
            else:
                text += line

        if self.config.shuffle_facts_seed > 0:
            self.data = [self.data[i] for i in rnd_idx(N=len(self.data), seed=self.config.shuffle_facts_seed)]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "start": 0.0,
                    "end": 0.90
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "start": 0.90,
                    "end": 1.00
                },
            )
        ]
