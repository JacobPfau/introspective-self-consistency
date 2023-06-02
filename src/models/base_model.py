from enum import Enum

INVALID_RESPONSE = "INVALID_RESPONSE"


class BaseModel(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @property
    def invalid_response(self):
        return INVALID_RESPONSE
