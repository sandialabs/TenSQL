from typing import Generator, Optional, Dict, Any, Tuple
from ..util import GBTensor

from .Node import Node

from ..Types import Type

class PrimaryKey(Node):
    def __init__(
        self, stencil: GBTensor, colnum: int, type_: Type, *, name: str
    ) -> None:
        super().__init__(name)
        self._type = type_
        self._stencil = stencil
        self._colnum = colnum

    @property
    def stencil(self) -> GBTensor:
        return self._stencil

    @property
    def colnum(self) -> int:
        return self._colnum

    @property
    def type_(self) -> Type:
        return self._type

    def __iter__(self) -> Generator[Tuple[str, Node], None, None]:
        yield ("stencil", self._stencil)

    @property
    def dsl(self) -> str:
        return (
            f"{type(self).__name__}(name={self._name!r}, stencil={self._stencil.dsl})"
        )

    @property
    def description(self) -> str:
        return f"{type(self).__name__}(name={self._name!r})"

    def get_attributes(self) -> Dict[str, Any]:
        return {
            "colnum": self._colnum,
            "stencil": self._stencil,
            "name": self._name,
            "type_": self._type,
        }
