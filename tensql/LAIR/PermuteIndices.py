from typing import Generator, Optional, Dict, Any, Tuple

from .Node import Node
from ..util import Axes


class PermuteIndices(Node):
    def __init__(
        self, operand: Node, new_axes: Optional[Axes], name: Optional[str] = None
    ) -> None:
        self._operand = operand
        if new_axes is None:
            self._new_axes = None
        else:
            self._new_axes = tuple(new_axes)
        super().__init__(name)

    @property
    def operand(self) -> Node:
        return self._operand

    @property
    def new_axes(self) -> Optional[Axes]:
        return self._new_axes

    def __iter__(self) -> Generator[Tuple[str, Node], None, None]:
        yield ("operand", self._operand)

    @property
    def description(self) -> str:
        return f"{type(self).__name__}({self._new_axes})"

    @property
    def dsl(self) -> str:
        return f"{type(self).__name__}({self.operand.dsl}, new_axes={self._new_axes})"

    def get_attributes(self) -> Dict[str, Any]:
        return {
            "operand": self._operand,
            "new_axes": self._new_axes,
            "name": self._name,
        }
