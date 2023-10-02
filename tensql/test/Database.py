import h5py
import pygraphblas
import pygraphblas.types

from ..util import export
from .test_util import register, TmpDirMixin
from .. import Column, Table, Database, Types

__all__ = []


@export
@register()
class TestDBTableInitNoStencil(TmpDirMixin):
    def runTest(self):
        shape = (10,)
        col_type = Types.BigInt()
        col_data = pygraphblas.Vector.sparse(col_type.as_pygrb, size=shape[0])
        col_name = "Data"
        column = Column(col_name, col_data, type_=col_type)

        pk_name, pk_data, pk_gbtype = "PK", None, Types.BigInt()
        pk_column = Column(pk_name, pk_data, type_=pk_gbtype)

        db = Database()
        table = db.create_table("tmp", shape, [pk_column, column])

        self.assertEqual(table.shape, shape)
        self.assertEqual(
            list(table.columns), [(pk_name, pk_column), (col_name, column)]
        )

@export
@register()
class TestTableFromScalar(TmpDirMixin):
    def runTest(self):
        col_type = Types.BigInt()
        col_data = pygraphblas.Scalar.from_type(col_type.as_pygrb)

        db = Database()
        table = db.create_table_from_tensor("tmp", col_data)

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
class TestTableFromVector(TmpDirMixin):
    def runTest(self):
        shape = (10,)
        col_type = Types.BigInt()
        col_data = pygraphblas.Vector.sparse(typ=col_type.as_pygrb, size=shape[0])

        db = Database()
        table = db.create_table_from_tensor("tmp", col_data)

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
class TestTableFromMatrix(TmpDirMixin):
    def runTest(self):
        shape = (10, 7)
        col_type = Types.BigInt()
        col_data = pygraphblas.Matrix.sparse(typ=col_type.as_pygrb, nrows=shape[0], ncols=shape[1])

        db = Database()
        table = db.create_table_from_tensor("tmp", col_data)

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
class TestTableFromDefinitionScalar(TmpDirMixin):
    def runTest(self):
        col_type = Types.BigInt()

        db = Database()
        table = db.create_table_from_definition(
            "tmp",
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
class TestTableFromDefinitionVector(TmpDirMixin):
    def runTest(self):
        shape = (2 ** 60,)
        col_type = Types.BigInt()

        db = Database()
        table = db.create_table_from_definition(
            "tmp",
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
class TestTableFromDefinitionMatrix(TmpDirMixin):
    def runTest(self):
        shape = (2 ** 60, 2 ** 60)
        col_type = Types.BigInt()

        db = Database()
        table = db.create_table_from_definition(
            "tmp",
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

        db = Database()
        table = db.create_table_from_tensor("tmp", col_data)
        
        data = [
            (4,),
        ]

        def test(db, N):
            db.save("tmp.h5")
            db2 = Database.load("tmp.h5")
            table2 = db2['tmp']

            self.assertEqual(table2.count(), N)
            self.assertEqual(len(table2), N)
            observed = list(table2)
            expected = [{"value":val} for (val,) in data[:N]]
            self.assertEqual(expected, observed)

        test(db, 0)
        table.insert(value=data[0][0])
        test(db, 1)

@export
@register()
class TestTableSaveLoadHDF5Vector(TmpDirMixin):
    def runTest(self):
        shape = (10,)
        col_type = Types.BigInt()
        col_data = pygraphblas.Vector.sparse(typ=col_type.as_pygrb, size=shape[0])

        db = Database()
        table = db.create_table_from_tensor("tmp", col_data)
        
        (N,) = shape
        data = [
            (0, 2),
            (N-1, 4)
        ]

        def test(db, N):
            db.save("tmp.h5")
            db2 = Database.load("tmp.h5")
            table2 = db2['tmp']

            self.assertEqual(table2.count(), N)
            self.assertEqual(len(table2), N)

            observed = list(table2)
            expected = [{"idx":idx, "value":val} for idx, val in data[:N]]
            self.assertEqual(expected, observed)

        test(db, 0)
        for it_count, (idx, val) in enumerate(data):
            table.insert(idx=idx, value=val)
            test(db, it_count+1)

@export
@register()
class TestTableSaveLoadHDF5Matrix(TmpDirMixin):
    def runTest(self):
        shape = (10, 7)
        col_type = Types.BigInt()
        col_data = pygraphblas.Matrix.sparse(typ=col_type.as_pygrb, nrows=shape[0], ncols=shape[1])

        db = Database()
        table = db.create_table_from_tensor("tmp", col_data)
        
        M, N = shape
        data = [
            (0, 0, 2),
            (0, N-1, 4),
            (M-1, 0, 8),
            (M-1, N-1, 16)
        ]

        def test(db, N):
            db.save("tmp.h5")
            db2 = Database.load("tmp.h5")
            table2 = db2['tmp']

            self.assertEqual(table2.count(), N)
            self.assertEqual(len(table2), N)

            observed = list(table2)
            expected = [{"ridx":ridx, "cidx":cidx, "value":val} for ridx, cidx, val in data[:N]]
            self.assertEqual(expected, observed)

        test(db, 0)
        for it_count, (ridx, cidx, val) in enumerate(data):
            table.insert(ridx=ridx, cidx=cidx, value=val)
            test(db, it_count+1)
