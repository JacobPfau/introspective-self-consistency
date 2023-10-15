from dataclasses import dataclass, field, fields
from typing import Dict, List


@dataclass
class AmbiBenchConfig:

    construction_format: str
    n_shots: int
    n_queries: int
    n_multiple_choices: int
    prob_of_ambiguous: float

    needs_instruction: bool = False
    needs_informative: bool = False
    no_salient_task: bool = False
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

    candidate_categories: List[str] = field(
        default_factory=list,
        metadata={"help": "List of possible categories that could generate examples."},
    )

    assistance_prompts: Dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Additional prompts for clarification, verbalisation"},
    )

    @classmethod
    def from_dict(cls, params):
        class_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in class_fields})

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = AmbiBenchConfig(**self.config)
