"""Loader registry — get_loader(name) returns the right adapter."""
from .code_review import CodeReviewLoader
from .funcom import FuncomLoader
from .tesoro import TesoroLoader
from .codereval import CoderEvalLoader
from .robustness_copilot import RobustnessCopilotLoader
from .dome import DomeLoader
from .pentacet import PentacetLoader

_REGISTRY = {
    "code_review": CodeReviewLoader,
    "funcom": FuncomLoader,
    "tesoro": TesoroLoader,
    "codereval": CoderEvalLoader,
    "robustness_copilot": RobustnessCopilotLoader,
    "dome": DomeLoader,
    "pentacet": PentacetLoader,
}

def get_loader(name, raw_path, language=None):
    if name not in _REGISTRY:
        raise KeyError(f"No loader for '{name}'. Known: {list(_REGISTRY)}")
    return _REGISTRY[name](raw_path, language)
