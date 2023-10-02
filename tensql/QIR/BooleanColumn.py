from .ABCBooleanExpression import ABCBooleanExpression
from .QirColumn import QirColumn


class BooleanColumn(QirColumn, ABCBooleanExpression):
    pass
