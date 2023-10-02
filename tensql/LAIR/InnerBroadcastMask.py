from typing import Generator, Optional, Tuple, Dict, Any, Sequence

from ..util import EncoderDecoder, SequentialIdGenerator

from .DotMxM import DotMxM

from .Node import Node
from .Mask import Mask
from .ElementwiseMultiply import ElementwiseMultiply
from .CastLike import CastLike
from .PermuteIndices import PermuteIndices
from .ElementwiseApply import ElementwiseApplyBindSecond
from .VectorToDiagMatrix import VectorToDiagMatrix

AxesSpecifier = Sequence[int]

class InnerBroadcastMask(Node):
    def __init__(
        self,
        left: Node,
        left_indices: AxesSpecifier,
        right: Node,
        right_indices: AxesSpecifier,
        out_indices: AxesSpecifier,
        name: Optional[str] = None,
    ) -> None:
        self._left = left
        self._right = right

        idx_re_encoder = EncoderDecoder(SequentialIdGenerator())
        self._out_indices = tuple(idx_re_encoder[idx] for idx in out_indices)
        self._left_indices = tuple(idx_re_encoder[idx] for idx in left_indices)
        self._right_indices = tuple(idx_re_encoder[idx] for idx in right_indices)

        super().__init__(name)

    @property
    def op(self) -> str:
        return "FIRST"

    @property
    def left(self) -> Node:
        return self._left

    @property
    def left_indices(self) -> Tuple[int, ...]:
        return self._left_indices

    @property
    def right(self) -> Node:
        return self._right

    @property
    def right_indices(self) -> Tuple[int, ...]:
        return self._right_indices

    @property
    def out_indices(self) -> Tuple[int, ...]:
        return self._out_indices

    @property
    def pattern(self) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        return (self._left_indices, self._right_indices, self._out_indices)

    def __iter__(self) -> Generator[Tuple[str, Node], None, None]:
        yield ("left", self._left)
        yield ("right", self._right)

    @property
    def dsl(self) -> str:
        return f"{type(self).__name__}({self._left.dsl}, {self._left_indices}, {self._right.dsl}, {self._right_indices}, {self._out_indices})"

    @property
    def description(self) -> str:
        return f"{type(self).__name__}({self.pattern})"

    def get_attributes(self) -> Dict[str, Any]:
        return {
            "left": self._left,
            "left_indices": self._left_indices,
            "right": self._right,
            "right_indices": self._right_indices,
            "out_indices": self._out_indices,
            "name": self._name,
        }


