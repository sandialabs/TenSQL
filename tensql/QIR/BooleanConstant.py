from typing import Optional
from .Constant import Constant
from .ABCBooleanExpression import ABCBooleanExpression


class BooleanConstant(Constant, ABCBooleanExpression):
    def __init__(self, value: bool, name: Optional[str] = None) -> None:
        self._value = value
        super().__init__(value=value, name=name)

    @property
    def value(self) -> bool:
        return self._value


class QirFalse(BooleanConstant):
    def __init__(self, name: Optional[str] = None) -> None:
        Constant.__init__(self=self, value=False, name=name)


class QirTrue(BooleanConstant):
    def __init__(self, name: Optional[str] = None) -> None:
        Constant.__init__(self=self, value=True, name=name)
