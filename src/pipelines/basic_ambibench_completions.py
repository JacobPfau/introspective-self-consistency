"""This module provides methods to load AmbiBench examples
TODO:
    integrate interface to generate AmbiBench-style exaples directly from here.
    Requires submodules or importing code from `task_ambiguity` repo
"""

import json
from pathlib import Path
from typing import Union

from src.structures.ambibench import AmbiBenchDataset


def load_ambibench_dataset(json_path: Union[str, Path]) -> AmbiBenchDataset:

    with open(json_path) as f_in:
        ambibench_data = json.load(f_in)

    dataset = AmbiBenchDataset.from_dict(ambibench_data)
    return dataset
