from typing import Generator, Optional, Dict, Any, Tuple

from .Node import Node


class Output(Node):
    def __init__(self, operand: Node, name: str) -> None:
        self._operand = operand
        super().__init__(name)

    @property
    def operand(self) -> Node:
        return self._operand

    def __iter__(self) -> Generator[Tuple[str, Node], None, None]:
        yield ("operand", self._operand)

    @property
    def description(self):
        return f"{type(self).__name__}({self.name!r})"

    @property
    def dsl(self) -> str:
        return f"{type(self).__name__}({self.operand.dsl}, name={self.name!r})"

    def get_attributes(self) -> Dict[str, Any]:
        return {"operand": self._operand, "name": self._name}
