from typing import Generator, Optional, Tuple, Dict, Any

from ..util import EncoderDecoder, SequentialIdGenerator

from .DotMxM import DotMxM

from .Node import Node
from .Mask import Mask
from .ElementwiseMultiply import ElementwiseMultiply
from .CastLike import CastLike
from .PermuteIndices import PermuteIndices
from .ElementwiseApply import ElementwiseApplyBindSecond
from .VectorToDiagMatrix import VectorToDiagMatrix

AxesSpecifier = Tuple[int, ...]

class Promote(Node):
    def __init__(
        self,
        operand: Node,
        operand_indices: AxesSpecifier,
        out_indices: AxesSpecifier,
        name: Optional[str] = None,
    ) -> None:
        self._operand = operand

        idx_re_encoder = EncoderDecoder(SequentialIdGenerator())
        self._operand_indices = tuple(idx_re_encoder[idx] for idx in operand_indices)
        self._out_indices = tuple(idx_re_encoder[idx] for idx in out_indices)

        super().__init__(name)

    @property
    def operand(self) -> Node:
        return self._operand

    @property
    def signature(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return (self._operand_indices, self._out_indices)

    @property
    def operand_indices(self) -> Node:
        return self._operand_indices

    @property
    def out_indices(self) -> Node:
        return self._out_indices

    def __iter__(self) -> Generator[Tuple[str, Node], None, None]:
        yield ("operand", self._operand)

    @property
    def dsl(self) -> str:
        return f"{type(self).__name__}({self.operand.dsl}, {self._operand_indices}, {self._out_indices})"

    @property
    def description(self) -> str:
        return f"{type(self).__name__}({self._operand_indices}, {self._out_indices})"

    def get_attributes(self) -> Dict[str, Any]:
        return {
            "operand": self._operand,
            "operand_indices": self._operand_indices,
            "out_indices": self._out_indices,
            "name": self._name,
        }


def OptimizedPromote(
    operand: Node,
    operand_indices: AxesSpecifier,
    mask: Node,
    mask_indices: AxesSpecifier,
) -> Node:
    assert set(operand_indices) - set(mask_indices) == set()

    return Mask(Promote(operand, operand_indices, mask_indices), mask)
