from .locoop import LoCoOp
from .sct import SCT
from .mambo import Mambo


from .coco import LoCoOpCoCo, SCTCoCo, MamboCoCo

__all__ = [
    # Foreground-background decomposition methods
    "LoCoOp",
    "SCT",
    "Mambo",
    # + CoCo
    "LoCoOpCoCo",
    "SCTCoCo",
    "MamboCoCo",
]