from enum import Enum


class BaseModel(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
