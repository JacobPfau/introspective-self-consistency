from dataclasses import dataclass, fields
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

        mandatory_fields = [field.name for field in fields(cls) if field.init]
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
    sequence_type: str = "integer"
    num_samples: int = 1
    on_ambiguous_sequences: bool = True
    num_shots: int = 4
    shot_method: str = "random"
    task_prompt: str = "self-consistency"
    role_prompt: Optional[str] = None


@dataclass
class Q12LogprobInequalityConfig(BaseEvalConfig):
    csv_input_path: str
    num_shots = 4
    num_valid = 2
    num_invalid = 3
    cot = False
    few_shot_prompt_type = "random"
    invalid_fn_type = "random"
