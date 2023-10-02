import sys
import gc

import numpy as np

from .common import CommonDatabaseUnitTest
from ..test_util import register
from ...util import export
from ... import QIR
from ... import Types

__all__ = []

@export
@register()
class TestString_insert(CommonDatabaseUnitTest):
    def runTest(self):
        qdb = QIR.QirDatabase(self.db)

        strvec = self.db.create_table_from_definition(
            "StrVec",
            {
                "idx": Types.BigInt(),
                "value": Types.Text()
            },
            ["idx"]
        )

        input_rows = [
            dict(idx=1, value="sdkfjkel"),
            dict(idx=2, value="sadfewewqw"),
            dict(idx=3, value="asdfewqwwe")
        ]
        self.data = input_rows

        result = (
            qdb.insert(qdb.StrVec)
              .values(input_rows)
              .run()
        )

        self.assertEqual(len(input_rows), result.count())

        rows = list(result)

        self.assertEqual(input_rows, rows)

@export
@register()
class TestString_query_result(TestString_insert):
    def setUp(self):
        super().setUp()
        super().runTest()

    def runTest(self):
        qdb = QIR.QirDatabase(self.db)
        result = (
            qdb.query(qdb.StrVec)
              .select(qdb.StrVec.idx, qdb.StrVec.value)
              .run()
        )
        self.assertEqual(
          len(self.data),
          result.count()
        )
        self.assertEqual(
            self.data,
            list(result)
        )

@export
@register()
class TestString_where_expression_equals(TestString_insert):
    def setUp(self):
        super().setUp()
        super().runTest()

    def runTest(self):
        qdb = QIR.QirDatabase(self.db)
        data = self.data
        for row in data:
          idx, value = row['idx'], row['value']
          result = (
              qdb.query(qdb.StrVec)
                .select(qdb.StrVec.idx, qdb.StrVec.value)
                .where(qdb.StrVec.value == value)
                .run()
          )
          self.assertEqual(
              sorted([tuple(row.items()) for row in data if row['value'] == value]),
              sorted(tuple(row.items()) for row in result)
          )

@export
@register()
class TestString_where_expression_notequals(TestString_insert):
    def setUp(self):
        super().setUp()
        super().runTest()

    def runTest(self):
        qdb = QIR.QirDatabase(self.db)
        data = self.data
        for row in data:
          idx, value = row['idx'], row['value']
          result = (
              qdb.query(qdb.StrVec)
                .select(qdb.StrVec.idx, qdb.StrVec.value)
                .where(qdb.StrVec.value != value)
                .run()
          )
          self.assertEqual(
              sorted([tuple(row.items()) for row in data if row['value'] != value]),
              sorted(tuple(row.items()) for row in result)
          )
          
