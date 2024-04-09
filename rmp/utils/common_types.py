import os
from collections.abc import Sequence
from typing import Any, Callable, Tuple, TypeVar, Union

T = TypeVar("T")

# generic
PathLike = Union[os.PathLike, str]
Function = Callable[..., Any]

# numbers
Number = Union[int, float]
PositiveNumber = Number

# sequences
MaybeSequence = Union[T, Sequence[T]]

IntTuple = Tuple[int, ...]
FloatTuple = Tuple[float, ...]

IntTuple2D = Tuple[int, int]
FloatTuple2D = Tuple[float, float]
SlicingTuple2D = Tuple[slice, slice]

IntTuple3D = Tuple[int, int, int]
FloatTuple3D = Tuple[float, float, float]
SlicingTuple3D = Tuple[slice, slice, slice]
