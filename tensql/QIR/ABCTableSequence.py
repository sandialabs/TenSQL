import abc
from typing import Generator

from .ABCNode import ABCNode
from .Table import QirTable

from .. import Database


class ABCTableSequence(ABCNode):
    @property
    @abc.abstractmethod
    def tables(self) -> Generator[QirTable, None, None]:
        pass

    @property
    @abc.abstractmethod
    def db(self) -> Database:
        pass
