from dataclasses import dataclass, field, fields
from typing import Dict, List


@dataclass
class AmbiBenchConfig:

    construction_format: str
    n_shots: int
    n_queries: int
    prob_of_ambiguous: float

    needs_instruction: bool = False
    needs_informative: bool = False
    include_ambiguous_examples: bool = False
    construction_types: List[str] = field(
        default_factory=list,
        metadata={"help": "List of tasks or categories for which to generate examples"},
    )

    # model: str

    @classmethod
    def from_dict(cls, params):
        class_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in class_fields})


@dataclass
class AmbiBenchDataset:

    date: str
    config: AmbiBenchConfig
    examples: List[Dict[str, str]] = field(
        default_factory=list, metadata={"help": "List of query-completion tuple"}
    )

    @classmethod
    def from_dict(cls, params):
        class_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in class_fields})

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = AmbiBenchConfig.from_dict(self.config)
