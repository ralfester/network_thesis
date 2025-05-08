from enum import Enum, auto


# CRIMINAL STATUS ENUM
class CriminalStatus(Enum):
    NON_CRIMINAL = auto()
    PETTY_CRIMINAL = auto()
    ORGANIZED_CRIMINAL = auto()
    VORY = auto()
