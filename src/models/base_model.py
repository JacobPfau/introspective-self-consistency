from enum import Enum
from typing import List

INVALID_RESPONSE = "INVALID_RESPONSE"


class BaseModel(Enum):
    @classmethod
    def list(cls) -> List[str]:
        return list(map(lambda c: c.value, cls))

    @property
    def invalid_response(self):
        return INVALID_RESPONSE
