from dataclasses import Field, _DataclassParams
from typing import Dict, Tuple, TypeVar

from typing_extensions import Literal, Protocol

__all__ = [
    "Importable",
    "DataClass",
    "Instantiable",
    "Just",
    "Builds",
    "PartialBuilds",
]


# TODO: Define Instantiable

T = TypeVar("T")


class _Importable(Protocol):
    __module__: str
    __name__: str


Importable = TypeVar("Importable")


class DataClass(Protocol):
    # We are doing something a bit hacky here.. but instead of
    # using, e.g., Type[Builds] everywhere, it is nicer to use `Builds`.
    # Thus we add this __call__ method to make it look like objects
    # of type `Build` (which would be an *instance* of Build) are
    # instantiable
    def __call__(self, *args, **kwargs) -> "DataClass":  # pragma: no cover
        ...

    __dataclass_fields__: Dict[str, Field]
    __dataclass_params__: _DataclassParams
    __mro__: Tuple[type, ...]
    __name__: str
    __qualname__: str
    __module__: str


class Instantiable(DataClass, Protocol[T]):  # pragma: no cover
    _target_: str


class Just(Instantiable, Protocol[T]):
    obj: str  # interpolated string for importing obj
    _target_: str = "raiden.hydra_utils.identity"


class Builds(Instantiable, Protocol[T]):  # pragma: no cover
    _convert_: Literal["none", "partial", "all"]
    _recursive_: bool


class PartialBuilds(Builds, Protocol[T]):
    _partial_target_: str
