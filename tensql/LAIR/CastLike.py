from typing import Generator, Optional, Dict, Any, Tuple

from .Node import Node


class CastLike(Node):
    def __init__(self, operand: Node, like: Node, name: Optional[str] = None) -> None:
        self._operand = operand
        self._like = like
        super().__init__(name)

    @property
    def operand(self) -> Node:
        return self._operand

    @property
    def like(self) -> Node:
        return self._like

    def __iter__(self) -> Generator[Tuple[str, Node], None, None]:
        yield ("operand", self._operand)
        yield ("like", self._like)

    def get_attributes(self) -> Dict[str, Any]:
        return {"operand": self._operand, "like": self._like, "name": self._name}

    @property
    def dsl(self) -> str:
        return f"{type(self).__name__}({self._operand.dsl}, {self._like.dsl})"
