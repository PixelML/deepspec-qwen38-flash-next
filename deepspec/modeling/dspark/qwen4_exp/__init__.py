# Qwen4Exp (Qwen3.8-Flash-Next) DSpark support reuses Qwen3DSparkModel as its
# draft architecture unchanged -- only config construction (this package's
# config.py) is Qwen4Exp/HC-specific. See config.py for the rationale.
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel

from .config import build_draft_config

__all__ = [
    "Qwen3DSparkModel",
    "build_draft_config",
]
