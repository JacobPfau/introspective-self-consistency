from dataclasses import dataclass


@dataclass
class SequenceType:
    fn_name: str
    base_fn: str


@dataclass
class ArithmeticProgression(SequenceType):
    fn_name: str = "arithmetic"
    base_fn: str = "lambda x: {} + {} * x"


@dataclass
class GeometricProgression(SequenceType):
    fn_name: str = "geometric"
    base_fn: str = "lambda x: {} * {} ** x"


@dataclass
class ExponentialProgression(SequenceType):
    fn_name: str = "exponential"
    base_fn: str = "lambda x: ({} * x) ** {}"


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
        return eval(self.__str__)(step + self.offset)


# Needs
# - list of all sequence types for prompt
