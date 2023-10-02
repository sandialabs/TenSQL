from typing import Generator, Optional, Tuple, Dict, Any
from ..util import EncoderDecoder, SequentialIdGenerator

from .DotMxM import DotMxM

from .Node import Node
from .Mask import Mask
from .Pattern import Pattern
from .Promote import Promote
from .CastLike import CastLike
from .PermuteIndices import PermuteIndices
from .ElementwiseApply import ElementwiseApplyBindSecond
from .VectorToDiagMatrix import VectorToDiagMatrix
from .ElementwiseMultiply import ElementwiseMultiply
from .VectorToColumnMatrix import VectorToColumnMatrix
from .VectorToRowMatrix import VectorToRowMatrix

AxesSpecifier = Tuple[int, ...]

# class InnerBroadcastStencil(Node):
#   def __init__(
#       self,
#       left: Node, left_indices: AxesSpecifier,
#       right: Node, right_indices: AxesSpecifier,
#       out_indices: AxesSpecifier,
#       name: Optional[str] = None
#   ) -> None:
#     self._left = left
#     self._left_indices = left_indices
#     self._right = right
#     self._right_indices = right_indices
#     self._out_indices = out_indices
#     super().__init__(name)

#   @property
#   def left(self) -> Node:
#     return self._left

#   @property
#   def left_indices(self) -> AxesSpecifier:
#     return self._left_indices

#   @property
#   def right(self) -> Node:
#     return self._right

#   @property
#   def right_indices(self) -> AxesSpecifier:
#     return self._right_indices

#   @property
#   def out_indices(self) -> AxesSpecifier:
#     return self._out_indices

#   def __iter__(self) -> Generator[Node, None, None]:
#     yield self._left
#     yield self._right

#   @property
#   def dsl(self) -> str:
#     return f"{type(self).__name__}({self.left.dsl}, {self._left_indices}, {self.right.dsl}, {self._right_indices}, {self._out_indices})"

#   @property
#   def description(self) -> str:
#     return f"{type(self).__name__}({self._left_indices}, {self._right_indices}, {self._out_indices})"

#   def get_attributes(self) -> Dict[str, Any]:
#     return {
#       'left': self._left,
#       'left_indices': self._left_indices,
#       'right': self._right,
#       'right_indices': self._right_indices,
#       'out_indices': self._out_indices,
#       'name': self._name
#     }


def OptimizedInnerBroadcastStencil(
    left: Node,
    left_indices: AxesSpecifier,
    right: Node,
    right_indices: AxesSpecifier,
    out_indices: AxesSpecifier,
) -> Node:
    assert set(left_indices) - set(left_indices) == set()

    idx_re_encoder = EncoderDecoder(SequentialIdGenerator())
    left_indices = tuple(idx_re_encoder[idx] for idx in left_indices)
    right_indices = tuple(idx_re_encoder[idx] for idx in right_indices)
    out_indices = tuple(idx_re_encoder[idx] for idx in out_indices)
    idx_desc = (left_indices, right_indices, out_indices)

    return Mask(
        Promote(left, left_indices, out_indices),
        Promote(right, right_indices, out_indices),
    )


#  return ElementwiseMultiply(
#    'LAND',
#    Promote(left, left_indices, out_indices),
#    Promote(right, right_indices, out_indices)
#  )
# return InnerBroadcastStencil(left, left_indices, right, right_indices, out_indices)

#  NODIM = tuple()
#
#  left = Pattern(left)
#  right = Pattern(right)
#
#  if idx_desc == (NODIM, NODIM, NODIM):
#    sO = ElementwiseMultiply("AND", left, right)
#
#  elif idx_desc == ((0,), NODIM, (0,)):
#    sO = ElementwiseApplyBindSecond("AND", left, right)
#  elif idx_desc == ((0,1), NODIM, (0,1)):
#    sO = ElementwiseApplyBindSecond("AND", left, right)
#  elif idx_desc == ((0,1), NODIM, (1,0)):
#    sO = ElementwiseApplyBindSecond("AND", PermuteIndices(left), right)
#
#  elif idx_desc == ((0,), (0,), (0,)):
#    sO = ElementwiseMultiply("AND", left, right)
#  elif idx_desc == ((0,), (1,), (0,1)):
#    mA = VectorToColumnMatrix(left)
#    mB = VectorToRowMatrix(right)
#    sO = DotMxM(mA, mB)
#  elif idx_desc == ((0,), (1,), (1,0)):
#    mA = VectorToRowMatrix(left)
#    mB = VectorToColumnMatrix(right)
#    sO = DotMxM(mB, mA)
#
#  elif idx_desc == ((0,1), (0,), (0,1)):
#    mB = VectorToDiagMatrix(right)
#    sO = DotMxM(mB, left)
#  elif idx_desc == ((0,1), (0,), (1,0)):
#    mB = VectorToDiagMatrix(right)
#    sO = DotMxM(PermuteIndices(left), mB)
#  elif idx_desc == ((0,1), (1,), (0,1)):
#    mB = VectorToDiagMatrix(right)
#    sO = DotMxM(left, mB)
#  elif idx_desc == ((0,1), (1,), (1,0)):
#    mB = VectorToDiagMatrix(right)
#    sO = DotMxM(mB, PermuteIndices(left))
#
#  elif idx_desc == ((0,1), (0,1), (0,1)):
#    sO = ElementwiseMultiply("AND", left, right)
#  elif idx_desc == ((0,1), (0,1), (1,0)):
#    sO = PermuteIndices(ElementwiseMultiply("AND", left, right))
#  elif idx_desc == ((0,1), (1,0), (0,1)):
#    sO = ElementwiseMultiply("AND", left, PermuteIndices(right))
#  elif idx_desc == ((0,1), (1,0), (1,0)):
#    sO = ElementwiseMultiply("AND", PermuteIndices(left), right)
#
#  else:
#    sO = InnerBroadcastStencil(left, left_indices, right, right_indices, out_indices)
#
#  return sO
