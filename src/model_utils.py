from dataclasses import dataclass
from enum import IntEnum

class TFType(IntEnum):
    SOX2 = 1
    NANOG = 2

class PartnerState(IntEnum):
    FREE = -1
    DANGLING_SOX2 = -2
    DANGLING_NANOG = -3

@dataclass(frozen=True)
class TranscriptionFactor:
    id: int
    name: str
    valency: int
    is_activator: bool = False

    @property
    def dangling_id(self) -> int:
        """
        Dynamically generates the dangling integer state.
        If ID is 1, dangling is -2. If ID is 2, dangling is -3.
        This allows infinite TFs without hardcoding negative numbers.
        """
        return -(self.id + 1)
