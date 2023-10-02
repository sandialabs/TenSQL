from __future__ import annotations

import abc
from typing import Optional, Generator, TYPE_CHECKING

from .ABCNode import ABCNode
from .. import Types

if TYPE_CHECKING:
    from .QirColumn import QirColumn

class ABCExpression(ABCNode):
    def __init__(self, name: Optional[str]) -> None:
        super().__init__(name)

    @property
    @abc.abstractmethod
    def description(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def columns(self) -> Generator[QirColumn]:
        pass
