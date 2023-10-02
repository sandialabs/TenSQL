from typing import Generator, Optional, Union, Tuple
from .ABCNode import ABCNode
from .ABCExpression import ABCExpression
from .QirColumn import QirColumn


class Constant(ABCExpression):
    def __init__(
        self, value: Union[int, float, bool], name: Optional[str] = None
    ) -> None:
        self._value = value

        super().__init__(name)

    @property
    def value(self) -> Union[int, float, bool]:
        return self._value

    @property
    def description(self) -> str:
        return repr(self._value)

    def __iter__(self) -> Generator[Tuple[str, ABCNode], None, None]:
        yield from []

    @property
    def columns(self) -> Generator[QirColumn, None, None]:
        pass
