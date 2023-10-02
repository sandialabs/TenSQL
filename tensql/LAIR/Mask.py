from typing import Generator, Optional, Dict, Any, Tuple

from .Node import Node


class Mask(Node):
    def __init__(self, operand: Node, mask: Node, name: Optional[str] = None) -> None:
        self._operand = operand
        self._mask = mask
        super().__init__(name)

    @property
    def operand(self) -> Node:
        return self._operand

    @property
    def mask(self) -> Node:
        return self._mask

    def __iter__(self) -> Generator[Tuple[str, Node], None, None]:
        yield ("operand", self._operand)
        yield ("mask", self._mask)

    @property
    def op(self) -> str:
        return "FIRST"

    @property
    def left(self) -> Node:
        return self._operand

    @property
    def right(self) -> Node:
        return self._mask

    def get_attributes(self) -> Dict[str, Any]:
        return {"operand": self._operand, "mask": self._mask, "name": self._name}

    @property
    def dsl(self) -> str:
        return f"Mask({self.operand.dsl}, {self.mask.dsl})"
