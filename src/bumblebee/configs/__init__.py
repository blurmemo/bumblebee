from typing import TYPE_CHECKING
from ..utils import _LazyModule


_import_structure = {
    "trainer_args": ["TrainArguments"],
}

if TYPE_CHECKING:
    from .trainer_args import TrainArguments
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__   # type: ignore
    )