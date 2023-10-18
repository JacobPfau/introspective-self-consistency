from dataclasses import dataclass, field, fields
from typing import Optional, Union

from omegaconf import DictConfig, OmegaConf

from src.evals.errors import InvalidConfigError
from src.models import BaseModel, get_model_from_string
from src.pipelines import ShotSamplingType

MAX_OFFSET = 8
NUM_SHOTS = 8


@dataclass
class BaseEvalConfig:
    task: str
    model: BaseModel
    few_shot_prompt_type: ShotSamplingType

    def __post_init__(self):
        if isinstance(self.model, str):
            self.model = get_model_from_string(self.model)

        if isinstance(self.few_shot_prompt_type, str):
            try:
                self.few_shot_prompt_type = ShotSamplingType(
                    self.few_shot_prompt_type.lower()
                )
            except ValueError:
                raise InvalidConfigError(
                    f"Invalid few shot prompt type: '{self.few_shot_prompt_type}'"
                )

    @classmethod
    def from_dict(cls, params: Union[dict, DictConfig]):
        if isinstance(params, DictConfig):
            params = OmegaConf.to_container(params)

        mandatory_fields = [field.name for field in fields(cls)]
        return cls(**{k: v for k, v in params.items() if k in mandatory_fields})


@dataclass
class SequenceCompletionCapabilityConfig(BaseEvalConfig):
    num_shots = NUM_SHOTS
    max_offset = MAX_OFFSET
    csv_input_path: str


@dataclass
class SequenceCompletionEqConfig(BaseEvalConfig):
    num_shots = NUM_SHOTS
    max_offset = MAX_OFFSET
    few_shot_prompt_type: ShotSamplingType = ShotSamplingType.RANDOM


@dataclass
class SequenceCompletionBaseChangeConfig(BaseEvalConfig):
    num_samples: int = 1
    on_ambiguous_sequences: bool = True
    num_shots: int = 4
    task_prompt: str = "self-consistency"
    role_prompt: Optional[str] = None
    base: int = 10
    seed: int = 21
    custom_sequences: bool = False
    max_constant_term_one: int = 5
    max_constant_term_two: int = 5
    num_steps_to_check: int = 2
    step_offsets: int = 5

    def __post_init__(self):
        super().__post_init__()
        if self.base not in [2, 10]:
            raise ValueError(f"Base must be 2 or 10; got {self.base}")


@dataclass
class Q21LogprobInequalityConfig(BaseEvalConfig):
    csv_input_path: str
    num_shots: int = field(default=4)
    num_valid: int = field(default=2)
    num_invalid: int = field(default=3)
    num_multiple_choices: int = field(default=5)
    invalid_fn_type: str = field(default="random")

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.invalid_fn_type, str):
            try:
                self.invalid_fn_type = ShotSamplingType(self.invalid_fn_type.lower())
            except ValueError:
                raise InvalidConfigError(
                    f"Invalid few shot prompt type: '{self.invalid_fn_type}'"
                )


@dataclass
class Q22ModelVerbalizationConfig(BaseEvalConfig):
    csv_input_path: str
    num_shots: int = field(default=4)
    max_considerations: int = field(default=5)
