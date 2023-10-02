import unittest

import numpy as np

from ..test_util import TmpDirMixin
from ... import Database, Types

class CommonDatabaseUnitTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.db = Database()

        self.matrix = self.db.create_table_from_definition(
            "Matrix",
            {
                "ridx": Types.BigInt(),
                "cidx": Types.BigInt(),
                "value": Types.Real()
            },
            ["ridx", "cidx"]
        )

        self.np_matrix = np.zeros(shape=(100, 100))
        for it in range(100):
            val = 1.0 / (it+1)
            self.matrix.insert(ridx=it, cidx=it, value=val)
            self.np_matrix[it, it] = 1.0 / (it+1)
            if(it > 0):
                val = 7.0 / (it + 1)
                self.matrix.insert(ridx=it, cidx=it-1, value=7.0 / (it + 1))
                self.np_matrix[it, it-1] = val

                val = 5.0 / (it + 1)
                self.matrix.insert(ridx=it-1, cidx=it, value=5.0 / (it + 1))
                self.np_matrix[it-1, it] = val

        self.left_vector = self.db.create_table_from_definition(
            "LeftVector",
            {
                "idx": Types.BigInt(),
                "value": Types.Real()
            },
            ["idx"]
        )
        self.np_left_vector = np.zeros(shape=(100,), dtype=np.float32)
        for it in range(100):
            if it % 3 != 0:
                val = 3.0 / (it+1)
                self.np_left_vector[it] = val
                self.left_vector.insert(idx=it, value=val)

        self.right_vector = self.db.create_table_from_definition(
            "RightVector",
            {
                "idx": Types.BigInt(),
                "value": Types.Real()
            },
            ["idx"]
        )
        self.np_right_vector = np.zeros(shape=(100,), dtype=np.float32)
        for it in range(100):
            if it % 5 != 0:
                val = 2.0 / (it+1)
                self.np_right_vector[it] = val
                self.right_vector.insert(idx=it, value=val)

