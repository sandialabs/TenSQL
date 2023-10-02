from typing import Union, Optional

from ..Table import Table
from ..Database import Database
from .Query import Query
from .SelectQuery import SelectQuery
from .InsertQuery import InsertQuery
from .Table import QirTable

# from . import Query, SelectQuery, QirTable
from .LAIRTranslator import QueryVisitor


class QirDatabase:
    def __init__(self, db: Database):
        self._db = db
        self._executor = QueryVisitor(db)

    @property
    def db(self):
        return self._db

    def __getattr__(self, name: str) -> QirTable:
        table = self._db.tables.get(name)
        if table is None:
            raise AttributeError()
        else:
            return QirTable(self.db, name, table)

    def import_table(self, name, table) -> QirTable:
        ret = QirTable(self.db, name, table)
        self._db.add_table(name, ret.raw)
        return ret

    def drop_table(self, name) -> None:
        del self._db[name]

    def query(
      self,
      table: Union[QirTable, Table],
      alias: Optional[str] = None
    ):
        if isinstance(table, Table):
            table = QirTable(self, None, table, alias=alias)
        elif alias is not None:
            table = table.aliased(alias)

        return SelectQuery(self, table)

    def insert(self, table: QirTable, **kwargs):
        return InsertQuery(self, table, **kwargs)

    def run(self, node: Query) -> Table:
        return self._executor.visit(node)
