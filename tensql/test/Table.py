import os
import unittest

import h5py
import numpy as np
import pygraphblas
import pygraphblas.types

from ..util import export
from .test_util import register, TmpDirMixin
from .. import Column, Table, Types

__all__ = []


@export
@register()
class TestTableInitNoStencil(unittest.TestCase):
    def runTest(self):
        shape = (10,)
        col_type = Types.BigInt()
        col_data = pygraphblas.Vector.sparse(col_type.as_pygrb, size=shape[0])
        col_name = "Data"
        column = Column(col_name, col_data, type_=col_type)

        pk_name, pk_data, pk_type = "PK", None, Types.BigInt()
        pk_column = Column(pk_name, pk_data, type_=pk_type)

        table = Table(shape, [pk_column, column])

        self.assertEqual(table.shape, shape)
        self.assertEqual(
            list(table.columns), [(pk_name, pk_column), (col_name, column)]
        )

@export
@register()
class TestTableFromScalar(unittest.TestCase):
    def runTest(self):
        col_type = Types.BigInt()
        col_data = pygraphblas.Scalar.from_type(col_type.as_pygrb)

        table = Table.from_tensor(col_data)

        [[col_name, col]] = list(table.columns)
        self.assertEqual(col_name, "value")
        #self.assertEqual(col.data, col_data)
        self.assertEqual(table.shape, tuple())
        self.assertEqual(table.ndim, 0)
        self.assertTrue(table.is_scalar)
        self.assertFalse(table.is_vector)
        self.assertFalse(table.is_matrix)
        self.assertEqual(table.primary_key_names, tuple())
        self.assertIs(table.get_column(col_name), col)
        self.assertEqual(table.count(), 0)
        self.assertEqual(len(table), 0)


@export
@register()
class TestTableFromVector(unittest.TestCase):
    def runTest(self):
        shape = (10,)
        col_type = Types.BigInt()
        col_data = pygraphblas.Vector.sparse(typ=col_type.as_pygrb, size=shape[0])

        table = Table.from_tensor(col_data)

        [[pk_name, pk], [col_name, col]] = list(table.columns)
        self.assertEqual(col_name, "value")
        #self.assertEqual(col.data, col_data)
        self.assertEqual(pk_name, "idx")
        self.assertEqual(table.shape, shape)
        self.assertEqual(table.ndim, 1)
        self.assertFalse(table.is_scalar)
        self.assertTrue(table.is_vector)
        self.assertFalse(table.is_matrix)
        self.assertTrue(table.is_primary_key(pk))
        self.assertEqual(table.primary_key_names, (pk_name,))
        self.assertIs(table.get_column(col_name), col)
        self.assertEqual(table.count(), 0)
        self.assertEqual(len(table), 0)

@export
@register()
class TestTableFromMatrix(unittest.TestCase):
    def runTest(self):
        shape = (10, 7)
        col_type = Types.BigInt()
        col_data = pygraphblas.Matrix.sparse(typ=col_type.as_pygrb, nrows=shape[0], ncols=shape[1])

        table = Table.from_tensor(col_data)

        [[pk1_name, pk1], [pk2_name, pk2], [col_name, col]] = list(table.columns)
        self.assertEqual(col_name, "value")
        #self.assertEqual(col.data, col_data)
        self.assertEqual(pk1_name, "ridx")
        self.assertEqual(pk2_name, "cidx")
        self.assertEqual(table.shape, shape)
        self.assertEqual(table.ndim, 2)
        self.assertFalse(table.is_scalar)
        self.assertFalse(table.is_vector)
        self.assertTrue(table.is_matrix)
        self.assertEqual(table.primary_key_names, (pk1_name, pk2_name))
        self.assertIs(table.get_column(col_name), col)
        self.assertEqual(table.count(), 0)
        self.assertEqual(len(table), 0)

@export
@register()
class TestTableInsertIntoScalar(unittest.TestCase):
    def runTest(self):
        col_type = Types.BigInt()
        col_data = pygraphblas.Scalar.from_type(typ=col_type.as_pygrb)

        table = Table.from_tensor(col_data)
        
        table = Table.from_definition(
            {
                "value": col_type,
                "value2": col_type
            },
            []
        )
        
        data = [
            (4,),
        ]

        table.insert(value=data[0][0])

        self.assertEqual(table.count(), 1)
        self.assertEqual(len(table), 1)

        observed = list(table)
        expected = [{"value":val, "value2": None} for (val,) in data]
        self.assertEqual(expected, observed)

@export
@register()
class TestTableInsertIntoVector(unittest.TestCase):
    def runTest(self):
        shape = (10,)
        col_type = Types.BigInt()
        col_data = pygraphblas.Vector.sparse(typ=col_type.as_pygrb, size=shape[0])

        table = Table.from_definition(
            {
                "idx": col_type,
                "value": col_type,
                "value2": col_type
            },
            ["idx"]
        )
        
        (N,) = shape
        data = [
            (0, 2),
            (5, 4),
            (N-1, 8)
        ]

        for it_count, (idx, val) in enumerate(data):
            table.insert(idx=idx, value=val)

            self.assertEqual(table.count(), it_count+1)
            self.assertEqual(len(table), it_count+1)
            self.assertEqual(table._columns['value2'].data.nvals, 0)

            observed = list(table)
            expected = [{"idx":idx, "value":val, "value2": None} for idx, val in data[:it_count+1]]
            self.assertEqual(expected, observed)

@export
@register()
class TestTableInsertIntoVector2(unittest.TestCase):
    def runTest(self):
        shape = (10,)
        col_type = Types.BigInt()
        col_data = pygraphblas.Vector.sparse(typ=col_type.as_pygrb, size=shape[0])

        table = Table.from_definition(
            {
                "idx": col_type,
                "value": col_type,
                "value2": col_type
            },
            ["idx"]
        )
        
        (N,) = shape
        data = [
            (0, 2),
            (5, 4),
            (N-1, 8)
        ]

        for it_count, (idx, val) in enumerate(data):
            table.insert(idx=idx, value=val)

            self.assertEqual(table.count(), it_count+1)
            self.assertEqual(len(table), it_count+1)
            self.assertEqual(table._columns['value2'].data.nvals, 0)

            observed = table.to_records()
            self.assertEqual(("idx", "value", "value2"), observed.dtype.names)

            expected = np.empty(shape=(it_count+1,), dtype=observed.dtype)
            for off, (idx, val) in enumerate(data[:it_count+1]):
                expected[off]["idx"] = idx
                expected[off]["value"] = val
                expected[off]["value2"] = None
            np.testing.assert_equal(expected, observed)


@export
@register()
class TestTableInsertIntoMatrix(unittest.TestCase):
    def runTest(self):
        shape = (10, 7)
        col_type = Types.BigInt()
        col_data = pygraphblas.Matrix.sparse(typ=col_type.as_pygrb, nrows=shape[0], ncols=shape[1])

        table = Table.from_definition(
            {
                "ridx": col_type,
                "cidx": col_type,
                "value": col_type,
                "value2": col_type
            },
            ["ridx", "cidx"]
        )
        
        M, N = shape
        data = [
            (0, 0, 2),
            (0, N-1, 4),
            (M-1, 0, 8),
            (M-1, N-1, 16)
        ]

        for it_count, (ridx, cidx, val) in enumerate(data):
            table.insert(ridx=ridx, cidx=cidx, value=val)

            self.assertEqual(table.count(), it_count+1)
            self.assertEqual(len(table), it_count+1)

            observed = list(table)
            expected = [{"ridx":ridx, "cidx":cidx, "value":val, "value2":None} for ridx, cidx, val in data[:it_count+1]]
            self.assertEqual(expected, observed)



@export
@register()
class TestTableFromDefinitionScalar(unittest.TestCase):
    def runTest(self):
        col_type = Types.BigInt()

        table = Table.from_definition(
            {
                "value": col_type
            },
            []
        )

        [[col_name, col]] = list(table.columns)
        self.assertEqual(col_name, "value")
        #self.assertEqual(col.data, col_data)
        self.assertEqual(table.shape, tuple())
        self.assertEqual(table.ndim, 0)
        self.assertTrue(table.is_scalar)
        self.assertFalse(table.is_vector)
        self.assertFalse(table.is_matrix)
        self.assertEqual(table.primary_key_names, tuple())
        self.assertIs(table.get_column(col_name), col)
        self.assertEqual(table.count(), 0)
        self.assertEqual(len(table), 0)

@export
@register()
class TestTableFromDefinitionVector(unittest.TestCase):
    def runTest(self):
        shape = (2 ** 60,)
        col_type = Types.BigInt()

        table = Table.from_definition(
            {
                "idx": col_type,
                "value": col_type
            },
            ["idx"]
        )

        [[pk_name, pk], [col_name, col]] = list(table.columns)
        self.assertEqual(col_name, "value")
        #self.assertEqual(col.data, col_data)
        self.assertEqual(pk_name, "idx")
        self.assertEqual(table.shape, shape)
        self.assertEqual(table.ndim, 1)
        self.assertFalse(table.is_scalar)
        self.assertTrue(table.is_vector)
        self.assertFalse(table.is_matrix)
        self.assertTrue(table.is_primary_key(pk))
        self.assertEqual(table.primary_key_names, (pk_name,))
        self.assertIs(table.get_column(col_name), col)
        self.assertEqual(table.count(), 0)
        self.assertEqual(len(table), 0)

@export
@register()
class TestTableFromDefinitionMatrix(unittest.TestCase):
    def runTest(self):
        shape = (2 ** 60, 2 ** 60)
        col_type = Types.BigInt()

        table = Table.from_definition(
            {
                "ridx": col_type,
                "cidx": col_type,
                "value": col_type
            },
            ["ridx", "cidx"]
        )

        [[pk1_name, pk1], [pk2_name, pk2], [col_name, col]] = list(table.columns)
        self.assertEqual(col_name, "value")
        #self.assertEqual(col.data, col_data)
        self.assertEqual(pk1_name, "ridx")
        self.assertEqual(pk2_name, "cidx")
        self.assertEqual(table.shape, shape)
        self.assertEqual(table.ndim, 2)
        self.assertFalse(table.is_scalar)
        self.assertFalse(table.is_vector)
        self.assertTrue(table.is_matrix)
        self.assertEqual(table.primary_key_names, (pk1_name, pk2_name))
        self.assertIs(table.get_column(col_name), col)
        self.assertEqual(table.count(), 0)
        self.assertEqual(len(table), 0)


@export
@register()
class TestTableSaveLoadHDF5Scalar(TmpDirMixin):
    def runTest(self):
        col_type = Types.BigInt()
        col_data = pygraphblas.Scalar.from_type(typ=col_type.as_pygrb)

        table = Table.from_definition(
            {
                "value": col_type,
                "value2": col_type
            },
            []
        )
        
        data = [
            (4,),
        ]

        def test(table, N):
            with h5py.File("tmp.h5", "w") as fout:
                table.save_hdf5(fout.create_group("tmp"))
            
            with h5py.File("tmp.h5", "r") as fin:
                table2 = Table.load_hdf5(fin['tmp'])

            self.assertEqual(table2.count(), N)
            self.assertEqual(len(table2), N)
            observed = list(table2)
            expected = [{"value":val, "value2": None} for (val,) in data[:N]]
            self.assertEqual(expected, observed)

        test(table, 0)
        table.insert(value=data[0][0])
        test(table, 1)

@export
@register()
class TestTableSaveLoadHDF5Vector(TmpDirMixin):
    def runTest(self):
        shape = (10,)
        col_type = Types.BigInt()
        col_data = pygraphblas.Vector.sparse(typ=col_type.as_pygrb, size=shape[0])

        table = Table.from_definition(
            {
                "idx": col_type,
                "value": col_type,
                "value2": col_type
            },
            ["idx"]
        )
        
        (N,) = shape
        data = [
            (0, 2),
            (N-1, 4)
        ]

        def test(table, N):
            with h5py.File("tmp.h5", "w") as fout:
                table.save_hdf5(fout.create_group("tmp"))
        
            with h5py.File("tmp.h5", "r") as fin:
                table2 = Table.load_hdf5(fin['tmp'])

            self.assertEqual(table2.count(), N)
            self.assertEqual(len(table2), N)

            observed = list(table2)
            expected = [{"idx":idx, "value":val, "value2":None} for idx, val in data[:N]]
            self.assertEqual(expected, observed)

        test(table, 0)
        for it_count, (idx, val) in enumerate(data):
            table.insert(idx=idx, value=val)
            test(table, it_count+1)

@export
@register()
class TestTableSaveLoadHDF5Matrix(TmpDirMixin):
    def runTest(self):
        shape = (10, 7)
        col_type = Types.BigInt()
        col_data = pygraphblas.Matrix.sparse(typ=col_type.as_pygrb, nrows=shape[0], ncols=shape[1])

        table = Table.from_definition(
            {
                "ridx": col_type,
                "cidx": col_type,
                "value": col_type,
                "value2": col_type
            },
            ["ridx", "cidx"]
        )
        
        M, N = shape
        data = [
            (0, 0, 2),
            (0, N-1, 4),
            (M-1, 0, 8),
            (M-1, N-1, 16)
        ]

        def test(table, N):
            with h5py.File("tmp.h5", "w") as fout:
                table.save_hdf5(fout.create_group("tmp"))
        
            with h5py.File("tmp.h5", "r") as fin:
                table2 = Table.load_hdf5(fin['tmp'])

            self.assertEqual(table2.count(), N)
            self.assertEqual(len(table2), N)

            observed = list(table2)
            expected = [{"ridx":ridx, "cidx":cidx, "value":val, "value2": None} for ridx, cidx, val in data[:N]]
            self.assertEqual(expected, observed)

        test(table, 0)
        for it_count, (ridx, cidx, val) in enumerate(data):
            table.insert(ridx=ridx, cidx=cidx, value=val)
            test(table, it_count+1)

@export
@register()
class TestTableAddScalar(TmpDirMixin):
    def runTest(self):
        col_type = Types.BigInt()
        col_data = pygraphblas.Scalar.from_type(typ=col_type.as_pygrb)

        table = Table.from_tensor(col_data)
        
        data = [
            (4,),
            (8,),
            (16,),
        ]

        def test(table, N):
            table2 = Table.from_tensor(col_data)
            for it, (val,) in enumerate(data[:N]):
                table.insert(value=val)
                table2.add(table, inplace=True)

            observed = list(table2)
            if N == 0:
                self.assertEqual(table2.count(), 0)
                self.assertEqual(len(table2), 0)
                expected = []
            else:
                self.assertEqual(table2.count(), 1)
                self.assertEqual(len(table2), 1)
                expected = [{"value":sum(val for (val,) in data[:N])}]
            self.assertEqual(expected, observed)

        test(table, 0)
        test(table, 1)
        test(table, 2)
        test(table, 3)



@export
@register()
class TestTableAddVector(TmpDirMixin):
    def runTest(self):
        shape = (10,)
        col_type = Types.BigInt()
        col_data = pygraphblas.Vector.sparse(typ=col_type.as_pygrb, size=shape[0])

        table = Table.from_tensor(col_data)
        
        (N,) = shape
        data = [
            (0, 2),
            (2, 4),
            (N-1, 8)
        ]

        def test(table, N):
            table2 = Table.from_tensor(col_data)
            table2.add(table, inplace=True)

            self.assertEqual(table2.count(), N)
            self.assertEqual(len(table2), N)

            observed = list(table2)
            expected = [{"idx":idx, "value":val} for idx, val in data[:N]]
            self.assertEqual(expected, observed)

        test(table, 0)
        for it_count, (idx, val) in enumerate(data):
            table.insert(idx=idx, value=val)
            test(table, it_count+1)

@export
@register()
class TestTableAddMatrix(TmpDirMixin):
    def runTest(self):
        shape = (10, 7)
        col_type = Types.BigInt()
        col_data = pygraphblas.Matrix.sparse(typ=col_type.as_pygrb, nrows=shape[0], ncols=shape[1])

        table = Table.from_tensor(col_data, copy=True)
        
        M, N = shape
        data = [
            (0, 0, 2),
            (0, N-1, 4),
            (M-1, 0, 8),
            (M-1, N-1, 16)
        ]

        def test(table, n):
            table2 = Table.from_tensor(col_data, copy=True)
            table2.add(table, inplace=True)

            self.assertEqual(table2.count(), n)
            self.assertEqual(len(table2), n)
            observed = list(table2)
            expected = [{"ridx":ridx, "cidx":cidx, "value":val} for ridx, cidx, val in data[:n]]
            self.assertEqual(expected, observed)

        test(table, 0)
        for it_count, (ridx, cidx, val) in enumerate(data):
            table.insert(ridx=ridx, cidx=cidx, value=val)
            test(table, it_count+1)

@export
@register()
class TestTableSaveLoadHDF5ScalarText(TmpDirMixin):
    def runTest(self):
        col_type = Types.Text()
        col_data = pygraphblas.Scalar.from_type(typ=col_type.as_pygrb)

        table = Table.from_definition(
            {
                "value": col_type,
                "value2": col_type
            },
            []
        )
        
        data = [
            ("Hello World",),
        ]

        def test(table, N):
            with h5py.File("tmp.h5", "w") as fout:
                table.save_hdf5(fout.create_group("tmp"))
            
            with h5py.File("tmp.h5", "r") as fin:
                table2 = Table.load_hdf5(fin['tmp'])

            self.assertEqual(table2.count(), N)
            self.assertEqual(len(table2), N)
            observed = list(table2)
            expected = [{"value":val, "value2": None} for (val,) in data[:N]]
            self.assertEqual(expected, observed)

        test(table, 0)
        table.insert(value=data[0][0])
        test(table, 1)
