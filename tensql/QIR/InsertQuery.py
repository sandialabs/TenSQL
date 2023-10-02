from typing import Optional, Sequence, Union, Generator, Tuple, Any, TYPE_CHECKING, Dict

import numpy as np

from .ABCTableSequence import ABCTableSequence
from .Query import Query
from .ABCNode import ABCNode
from .Table import QirTable
from ..Table import Table

ValuesType = Dict[str, Any]
RowValuesSequenceType = Sequence[ValuesType]

class InsertQuery(Query):
  def __init__(
    self,
    qirdb: Any,
    table: QirTable,
    *,
#    values: Optional[RowValuesSequenceType] = None,
    if_not_exists: bool = False,
#    from_table: Optional[QirTable] = None
  ):
    self._db = qirdb
    self._table = table
    self._if_not_exists = if_not_exists
    self._values = []
    self._from_records = None
    self._from_table = None
#    self._from_table = from_table

#    if values is None:
#        self._values = []
#    else:
#        self._values = list(values)

  @property
  def db(self) -> Any:
    return self._db

  @property
  def table(self) -> Table:
    return self._table

  def values(
    self,
    args: Optional[RowValuesSequenceType] = None,
    **kwargs: ValuesType
  ):
    assert args is None or len(kwargs) == 0
    assert self._from_table is None
    assert self._from_records is None

    if args is not None:
      self._values.extend(args)
    else:
      self._values.append(kwargs)

    return self

  def from_table(
    self,
    table: QirTable
  ):
    assert len(self._values) == 0
    assert self._from_table is None
    assert self._from_records is None
    self._from_table = table
    return self

  def from_records(
    self,
    data: np.ndarray
  ):
    assert len(self._values) == 0
    assert self._from_table is None
    assert self._from_records is None

    self._from_records = data
    return self

  def run(self) -> Table:
    ret = self._db.run(self)
    return ret

  def __iter__(self) -> Generator[Tuple[str, ABCNode], None, None]:
    yield ("input", self._table)
