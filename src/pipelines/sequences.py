from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SequenceType:
    fn_name: str
    base_fn: str


@dataclass
class ArithmeticProgression(SequenceType):
    fn_name: str = "arithmetic"
    base_fn: str = "lambda x: ({} * x) + {}"


@dataclass
class GeometricProgression(SequenceType):
    fn_name: str = "geometric"
    base_fn: str = "lambda x: ({} * x) * {}"


@dataclass
class ExponentialProgression(SequenceType):
    fn_name: str = "exponential"
    base_fn: str = "lambda x: ({} * x) ** {}"


@dataclass
class PowerProgression(SequenceType):
    fn_name: str = "power"
    base_fn: str = "lambda x: {} ** ({} * x)"


@dataclass
class BitwiseOrProgression(SequenceType):
    fn_name: str = "bitwise_or"
    base_fn: str = "lambda x: ({} * x) | {}"


@dataclass
class ModularProgression(SequenceType):
    fn_name: str = "modular"
    base_fn: str = "lambda x: (x * {}) % ({}+1)"
    # (x * 3) % (1+1)


@dataclass
class IndexingCriteriaProgression(SequenceType):
    fn_name: str = "indexing_criteria"
    base_fn: str = (
        "lambda x: [i for i in range(100) if i % ({} + 1) or i % ({} + 1)][x]"
    )


@dataclass
class RecursiveProgression(SequenceType):
    fn_name: str = "recursive"
    base_fn: str = "(lambda a:lambda v:a(a,v))(lambda fn,x:1 if x==0 else {} * x * fn(fn,x-1) + {})"


# @dataclass
# class LogarithmicProgression(SequenceType):
#     fn_name: str = "logarithmic"
#     base_fn: str = "lambda x: {} * log(x, {})"


# @dataclass
# class BitwiseAndProgression(SequenceType):
#     fn_name: str = "bitwise_and"
#     base_fn: str = "lambda x: ({} * x) & {}"


class IntegerSequence:
    sequence_type: SequenceType
    offset: int
    term_a: int
    term_b: int

    def __init__(
        self, sequence_type: SequenceType, offset: int, term_a: int, term_b: int
    ):
        self.sequence_type = sequence_type
        self.offset = offset
        self.term_a = term_a
        self.term_b = term_b

    def __str__(self) -> str:
        return self.sequence_type.base_fn.format(self.term_a, self.term_b)

    def roll_out(self, step: int) -> int:
        # TODO: decide whether to add the offset to step or add it to eval result
        # A) eval(fn)(step) + offset
        # B) eval(fn)(step + offset)
        return eval(str(self))(step + self.offset)


def get_all_sequences() -> List[SequenceType]:
    return [
        ArithmeticProgression,
        GeometricProgression,
        ExponentialProgression,
        PowerProgression,
        BitwiseOrProgression,
        ModularProgression,
        IndexingCriteriaProgression,
        RecursiveProgression,
    ]


def get_sequences_as_dict() -> Dict[str, str]:
    return {seq.fn_name: seq.base_fn for seq in get_all_sequences()}
