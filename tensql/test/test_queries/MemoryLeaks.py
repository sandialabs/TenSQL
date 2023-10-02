import base64
import gc
import os
import random
import sys
import time
import ctypes

import numpy as np
import psutil

from .common import CommonDatabaseUnitTest
from ..test_util import register
from ...util import export
from ... import QIR
from ... import Types

__all__ = []

def get_memory_usage():
  return psutil.Process(os.getpid()).memory_info().rss

@export
@register()
class TestMemoryLeaks_insert_float_values(CommonDatabaseUnitTest):
    @staticmethod
    def get_random_rows(num=1024*1024*4):
      return [dict(idx=it, value=random.random()) for it in range(num)]

    def runTest(self):
        qdb = QIR.QirDatabase(self.db)

        vec = self.db.create_table_from_definition(
            "vec",
            {
                "idx": Types.BigInt(),
                "value": Types.Real()
            },
            ["idx"]
        )
        
        gc.collect()
        ctypes.CDLL('libc.so.6').malloc_trim(0) 

        starting_memory = get_memory_usage()

        input_rows = self.get_random_rows()

        result = (
            qdb.insert(qdb.vec)
              .values(input_rows)
              .run()
        )
        self.assertEqual(len(input_rows), result.count())

        del input_rows

        self.assertGreater(get_memory_usage(), starting_memory + 1024*1024*4)
        del self.db['vec']
        del result
        gc.collect()
        ctypes.CDLL('libc.so.6').malloc_trim(0) 
        self.assertLess(get_memory_usage(), starting_memory + 1024*1024*4)

@export
@register()
class TestMemoryLeaks_insert_string_values(CommonDatabaseUnitTest):
    @staticmethod
    def get_random_strings(num_strings=1024*32, string_size = 1024*64):
      return [dict(idx=it, value=base64.b64encode(random.randbytes(string_size))) for it in range(num_strings)]

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

        gc.collect()
        ctypes.CDLL('libc.so.6').malloc_trim(0) 

        starting_memory = get_memory_usage()

        input_rows = self.get_random_strings()

        result = (
            qdb.insert(qdb.StrVec)
              .values(input_rows)
              .run()
        )
        self.assertEqual(len(input_rows), result.count())

        del input_rows

        self.assertGreater(get_memory_usage(), starting_memory + 1024*1024*1024)
        del self.db["StrVec"]
        del result
        gc.collect()
        ctypes.CDLL('libc.so.6').malloc_trim(0) 
        self.assertLess(get_memory_usage(), starting_memory + 1024*1024*4)


#@export
#@register()
#class TestMemoryLeaks_string_query_result(TestMemoryLeaks_string_insert):
#    def setUp(self):
#        super().setUp()
#        super().runTest()
#
#    def runTest(self):
#        qdb = QIR.QirDatabase(self.db)
#        result = (
#            qdb.query(qdb.StrVec)
#              .select(qdb.StrVec.idx, qdb.StrVec.value)
#              .run()
#        )
#        self.assertEqual(
#          len(self.data),
#          result.count()
#        )
#        self.assertEqual(
#            self.data,
#            list(result)
#        )
#
