from dataclasses import dataclass
from typing import Dict, List

from .sequences import SequenceType


@dataclass
class BinaryArithmeticProgression(SequenceType):
    fn_name: str = "bin_arithmetic"
    base_fn: str = "lambda x: bin(({} * x) + {})"


@dataclass
class BinaryGeometricProgression(SequenceType):
    fn_name: str = "bin_geometric"
    base_fn: str = "lambda x: bin(({} * x) + {})"


@dataclass
class BinaryExponentialProgression(SequenceType):
    fn_name: str = "bin_exponential"
    base_fn: str = "lambda x: bin(({} * x) ** {})"


@dataclass
class BinaryPowerProgression(SequenceType):
    fn_name: str = "bin_power"
    base_fn: str = "lambda x: bin({} ** ({} * x))"


@dataclass
class BinaryBitwiseOrProgression(SequenceType):
    fn_name: str = "bin_bitwise_or"
    base_fn: str = "lambda x: bin(({} * x) | {})"


@dataclass
class BinaryModularProgression(SequenceType):
    fn_name: str = "bin_modular"
    base_fn: str = "lambda x: bin((x * {}) % ({}+1))"
    # bin((x * 3) % (1+1))


@dataclass
class BinaryIndexingCriteriaProgression(SequenceType):
    fn_name: str = "bin_indexing_criteria"
    base_fn: str = (
        "lambda x: bin([i for i in range(100) if i % ({} + 1) or i % ({} + 1)][x])"
    )


@dataclass
class BinaryRecursiveProgression(SequenceType):
    fn_name: str = "bin_recursive"
    base_fn: str = "(lambda a:lambda v:bin(a(a,v)))(lambda fn,x:1 if x==0 else {} * x * fn(fn,x-1) + {})"


def get_all_binary_sequences() -> List[SequenceType]:
    return [
        BinaryArithmeticProgression,
        BinaryGeometricProgression,
        BinaryExponentialProgression,
        BinaryPowerProgression,
        BinaryBitwiseOrProgression,
        BinaryModularProgression,
        BinaryIndexingCriteriaProgression,
        BinaryRecursiveProgression,
    ]


def get_binary_sequences_as_dict() -> Dict[str, str]:
    return {seq.fn_name: seq.base_fn for seq in get_all_binary_sequences()}
