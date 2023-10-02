from typing import Generator, Optional, Dict, Any, Tuple

from .Node import Node


class ElementwiseMultiply(Node):
    def __init__(
        self, op: str, left: Node, right: Node, name: Optional[str] = None
    ) -> None:
        self._left = left
        self._right = right
        self._op = op
        if name is None:
            name = self.description
        super().__init__(name)

    @property
    def op(self) -> str:
        return self._op

    @property
    def left(self) -> Node:
        return self._left

    @property
    def right(self) -> Node:
        return self._right

    @property
    def description(self) -> str:
        return f"{type(self).__name__}({self._op!r})"

    @property
    def dsl(self) -> str:
        return f"({self._left.dsl} {self._op} {self._right.dsl})"

    def get_attributes(self) -> Dict[str, Any]:
        return {
            "left": self._left,
            "right": self._right,
            "op": self._op,
            "name": self._name,
        }

    def __iter__(self) -> Generator[Tuple[str, Node], None, None]:
        yield ("left", self._left)
        yield ("right", self._right)
