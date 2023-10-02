from typing import Generator, Optional, Sequence, Dict, Any, Tuple

from .Node import Node
from ..util import Axes


class Reduction(Node):
    def __init__(
        self, op: str, operand: Node, axes: Axes, name: Optional[str] = None
    ) -> None:
        self._operand = operand
        self._op = op
        self._axes = tuple(axes)
        super().__init__(name)

    @property
    def axes(self) -> Axes:
        return self._axes

    @property
    def operand(self) -> Node:
        return self._operand

    @property
    def op(self) -> str:
        return self._op

    def __iter__(self) -> Generator[Tuple[str, Node], None, None]:
        yield ("operand", self._operand)

    @property
    def description(self) -> str:
        return f"{type(self).__name__}({self._op!r}, axes={self.axes})"

    @property
    def dsl(self) -> str:
        return f"{type(self).__name__}({self.operand.dsl}, axes={self.axes})"

    def get_attributes(self) -> Dict[str, Any]:
        return {
            "op": self._op,
            "operand": self._operand,
            "axes": self._axes,
            "name": self._name,
        }


class Sum(Reduction):
    def __init__(self, operand: Node, axes: Axes, name: Optional[str] = None):
        super().__init__("+", operand, axes, name)


class Min(Reduction):
    def __init__(self, operand: Node, axes: Axes, name: Optional[str] = None):
        super().__init__("MIN", operand, axes, name)


class Max(Reduction):
    def __init__(self, operand: Node, axes: Axes, name: Optional[str] = None):
        super().__init__("MAX", operand, axes, name)
