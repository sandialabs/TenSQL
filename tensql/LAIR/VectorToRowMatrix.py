from typing import Generator, Optional, Dict, Any, Tuple

from .Node import Node


class VectorToRowMatrix(Node):
    def __init__(self, operand: Node, name: Optional[str] = None) -> None:
        self._operand = operand
        super().__init__(name)

    @property
    def operand(self) -> Node:
        return self._operand

    def __iter__(self) -> Generator[Tuple[str, Node], None, None]:
        yield ("operand", self._operand)

    @property
    def dsl(self) -> str:
        return f"{type(self).__name__}({self.operand.dsl})"

    def get_attributes(self) -> Dict[str, Any]:
        return {"operand": self._operand, "name": self._name}
