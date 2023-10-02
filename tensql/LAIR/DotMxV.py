from typing import Generator, Optional, Dict, Any, Tuple

from .Node import Node


class DotMxV(Node):
    def __init__(
        self,
        left: Node,
        right: Node,
        add_op: str,
        mul_op: str,
        name: Optional[str] = None,
    ) -> None:
        self._left = left
        self._right = right
        self._add_op = add_op
        self._mul_op = mul_op
        if name is None:
            name = self.description
        super().__init__(name)

    @property
    def left(self) -> Node:
        return self._left

    @property
    def right(self) -> Node:
        return self._right

    @property
    def add_op(self) -> str:
        return self._add_op

    @property
    def mul_op(self) -> str:
        return self._mul_op

    @property
    def dsl(self) -> str:
        return f"{type(self).__name__}({self._left.dsl}, {self._right.dsl}, {self._add_op!r}, {self._mul_op!r})"

    @property
    def description(self) -> str:
        return f"{type(self).__name__}({self._add_op!r}, {self._mul_op!r})"

    def get_attributes(self) -> Dict[str, Any]:
        return {"left": self._left, "right": self._right, "name": self._name}

    def __iter__(self) -> Generator[Tuple[str, Node], None, None]:
        yield ("left", self._left)
        yield ("right", self._right)
