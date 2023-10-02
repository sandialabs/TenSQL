from typing import Generator, Optional, Dict, Any, Tuple

from .Node import Node


class ElementwiseApply(Node):
    def __init__(self, op: str, operand: Node, name: Optional[str] = None) -> None:
        self._operand = operand
        self._op = op
        super().__init__(name)

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
        return f"{type(self).__name__}({self._op!r})"

    @property
    def dsl(self) -> str:
        return f"{type(self).__name__}({self._op!r}, {self._operand.dsl})"

    def get_attributes(self) -> Dict[str, Any]:
        return {"operand": self._operand, "op": self._op, "name": self._name}


class ElementwiseApplyBindFirst(Node):
    def __init__(
        self, op: str, operand: Node, scalar: Node, name: Optional[str] = None
    ) -> None:
        self._operand = operand
        self._scalar = scalar
        self._op = op
        super().__init__(name)

    @property
    def operand(self) -> Node:
        return self._operand

    @property
    def scalar(self) -> Node:
        return self._scalar

    @property
    def op(self) -> str:
        return self._op

    def __iter__(self) -> Generator[Tuple[str, Node], None, None]:
        yield ("operand", self._operand)
        yield ("scalar", self._scalar)

    @property
    def description(self) -> str:
        return f"{type(self).__name__}({self._op!r})"

    @property
    def dsl(self) -> str:
        return f"({self._operand.dsl} {self.op} {self._scalar.dsl})"

    def get_attributes(self) -> Dict[str, Any]:
        return {
            "operand": self._operand,
            "scalar": self._scalar,
            "op": self._op,
            "name": self._name,
        }


class ElementwiseApplyBindSecond(Node):
    def __init__(
        self, op: str, operand: Node, scalar: Node, name: Optional[str] = None
    ) -> None:
        self._operand = operand
        self._scalar = scalar
        self._op = op
        super().__init__(name)

    @property
    def operand(self) -> Node:
        return self._operand

    @property
    def scalar(self) -> Node:
        return self._scalar

    @property
    def op(self):
        return self._op

    def __iter__(self) -> Generator[Tuple[str, Node], None, None]:
        yield ("operand", self._operand)
        yield ("scalar", self._scalar)

    @property
    def description(self) -> str:
        return f"{type(self).__name__}({self._op!r})"

    @property
    def dsl(self) -> str:
        return f"({self._operand.dsl} {self.op} {self._scalar.dsl})"

    def get_attributes(self) -> Dict[str, Any]:
        return {
            "operand": self._operand,
            "scalar": self._scalar,
            "op": self._op,
            "name": self._name,
        }
