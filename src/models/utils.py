from enum import Enum

INVALID_RESPONSE = "INVALID_RESPONSE"


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
