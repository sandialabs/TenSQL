import abc
from typing import Any

from .ABCNode import ABCNode
from ..Database import Database


class Query(ABCNode):
    def __init__(self, db: Database):
        self._db = db

    @property
    def db(self) -> Database:
        return self._db

    @abc.abstractmethod
    def run(self) -> Any:
        pass
