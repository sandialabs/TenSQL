from typing import Generator, Optional, Dict, Any, Tuple
from ..util import GBTensor

from .Node import Node


class Tensor(Node):
    def __init__(
        self, tensor: GBTensor, stencil: Optional[GBTensor] = None, *, name: str
    ) -> None:
        self._tensor = tensor
        self._stencil = stencil
        super().__init__(name)

    @property
    def tensor(self) -> GBTensor:
        return self._tensor

    @property
    def stencil(self) -> Optional[GBTensor]:
        return self._stencil

    def __iter__(self) -> Generator[Tuple[str, Node], None, None]:
        if self._stencil is not None:
            yield ("stencil", self._stencil)
        else:
            yield from tuple()

    @property
    def dsl(self) -> str:
        if self._stencil is None:
            return f"{type(self).__name__}(name={self._name!r})"
        else:
            return f"{type(self).__name__}(name={self._name!r}, stencil={self._stencil.dsl})"

    @property
    def description(self) -> str:
        return f"{type(self).__name__}(name={self._name!r})"

    def get_attributes(self) -> Dict[str, Any]:
        return {"tensor": self._tensor, "stencil": self._stencil, "name": self._name}
