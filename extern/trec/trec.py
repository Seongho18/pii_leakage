import os

import datasets
import pandas as pd

from pii_leakage.extern.CustomBuilder import CustomBuilder
from pii_leakage.utils.random import rnd_idx


class CustomTrec(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")
    _DESCRIPTION = "A custom wrapper for the Enron dataset."
    _TEXT = "text"

    _PATH = {
        "path": "../dataset/trec"
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
            text = "".join(lines[:])
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

    def _generate_examples(self, split: str, start: float, end: float):
        """ Given a start and stop location, tag all PII and generate the dataset.
        We use multi_gpu generation for improved speed.
        """
        start_pos, end_pos = int(len(self.data) * start), int(len(self.data) * end)

        unique_identifier = start_pos
        for i, text in enumerate(self.data[start_pos:end_pos]):
            pseudonymized_text = text

            for _ in range(self.config.sample_duplication_rate):
                unique_identifier += 1
                yield f"{unique_identifier}", {
                    self._TEXT: pseudonymized_text
                }
