from typing import Generator, Optional, Dict, Any, Tuple

from .Node import Node

class Cast(Node):
    def __init__(self, operand: Node, type_: Any, name: Optional[str] = None) -> None:
        self._operand = operand
        self._type = type_
        super().__init__(name)

    @property
    def operand(self) -> Node:
        return self._operand

    @property
    def type_(self) -> Node:
        return self._type

    def __iter__(self) -> Generator[Tuple[str, Node], None, None]:
        yield ("operand", self._operand)

    def get_attributes(self) -> Dict[str, Any]:
        return {"operand": self._operand, "type_": self._type, "name": self._name}

    @property
    def dsl(self) -> str:
        return f"Cast({self.operand.dsl}, {self.type_.__name__})"
