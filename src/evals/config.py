from dataclasses import dataclass, field, fields
from typing import Optional, Union

from omegaconf import DictConfig, OmegaConf

from src.models.base_model import BaseModel
from src.models.utils import get_model_from_string

MAX_OFFSET = 8
NUM_SHOTS = 8


@dataclass
class BaseEvalConfig:
    task: str
    model: BaseModel

    def __post_init__(self):
        if isinstance(self.model, str):
            self.model = get_model_from_string(self.model)

    @classmethod
    def from_dict(cls, params: Union[dict, DictConfig]):
        if isinstance(params, DictConfig):
            params = OmegaConf.to_container(params)

        mandatory_fields = [field.name for field in fields(cls)]
        return cls(**{k: v for k, v in params.items() if k in mandatory_fields})


@dataclass
class AmbibenchCompletionConfig(BaseEvalConfig):
    data_glob: str


@dataclass
class AmbibenchCatPredConfig(BaseEvalConfig):
    data_glob: str
    multiple_choice: bool


@dataclass
class StringTransformationConfig(BaseEvalConfig):
    num_shots = NUM_SHOTS
    cot = False


@dataclass
class SequenceCompletionEqConfig(BaseEvalConfig):
    num_shots = NUM_SHOTS
    cot = True
    max_offset = MAX_OFFSET
    few_shot_prompt_type = "random"


@dataclass
class SequenceCompletionBaseChangeConfig(BaseEvalConfig):
    num_samples: int = 1
    on_ambiguous_sequences: bool = True
    num_shots: int = 4
    shot_method: str = "random"
    task_prompt: str = "self-consistency"
    role_prompt: Optional[str] = None
    _base: int = 10

    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, value):
        if value not in [2, 10]:
            raise ValueError("base can only be 2 or 10")
        self._base = value


@dataclass
class Q21LogprobInequalityConfig(BaseEvalConfig):
    csv_input_path: str
    num_shots: int = field(default=4)
    num_valid: int = field(default=2)
    num_invalid: int = field(default=3)
    num_multiple_choices: int = field(default=5)  # number of multiple choice options
    cot: bool = field(default=False)
    few_shot_prompt_type: str = field(default="random")
    invalid_fn_type: str = field(default="random")


@dataclass
class Q22ModelVerbalizationConfig(BaseEvalConfig):
    csv_input_path: str
    num_shots: int = field(default=4)
    max_considerations: int = field(default=5)
    few_shot_prompt_type: str = field(default="random")
