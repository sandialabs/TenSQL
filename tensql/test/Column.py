import unittest

import numpy as np
import pygraphblas
import pygraphblas.types

from ..util import export, ndim
from .test_util import register
from .. import Column, Types

__all__ = []

@export
@register()
class TestColumnScalarInit(unittest.TestCase):
    def runTest(self):
        type_ = Types.BigInt()
        data = pygraphblas.Scalar.from_type(type_.as_pygrb)
        name = "Test"
        column = Column(name, data, type_=type_)
        self.assertEqual(column.name, name)
        self.assertEqual(column.hidden, False)
        self.assertIs(column.data, data)
        self.assertIs(column.gbtype, type_.as_pygrb)
        self.assertTrue(column.is_scalar)
        self.assertFalse(column.is_vector)
        self.assertFalse(column.is_matrix)
        self.assertFalse(column.is_boolean)
        self.assertEqual(column.nptype, np.int64)
        self.assertEqual(ndim(column.data), 0)

@export
@register()
class TestColumnVectorInit(unittest.TestCase):
    def runTest(self):
        type_ = Types.BigInt()
        data = pygraphblas.Vector.sparse(type_.as_pygrb, size=10)
        name = "Test"
        column = Column(name, data, type_=type_)
        self.assertEqual(column.name, name)
        self.assertEqual(column.hidden, False)
        self.assertIs(column.data, data)
        self.assertIs(column.gbtype, type_.as_pygrb)
        self.assertFalse(column.is_scalar)
        self.assertTrue(column.is_vector)
        self.assertFalse(column.is_matrix)
        self.assertFalse(column.is_boolean)
        self.assertEqual(column.nptype, np.int64)
        self.assertEqual(ndim(column.data), 1)

@export
@register()
class TestColumnMatrixInit(unittest.TestCase):
    def runTest(self):
        type_ = Types.BigInt()
        data = pygraphblas.Matrix.sparse(type_.as_pygrb, nrows=10, ncols=10)
        name = "Test"
        column = Column(name, data, type_=type_)
        self.assertEqual(column.name, name)
        self.assertEqual(column.hidden, False)
        self.assertIs(column.data, data)
        self.assertIs(column.gbtype, type_.as_pygrb)
        self.assertFalse(column.is_scalar)
        self.assertFalse(column.is_vector)
        self.assertTrue(column.is_matrix)
        self.assertFalse(column.is_boolean)
        self.assertEqual(column.nptype, np.int64)
        self.assertEqual(ndim(column.data), 2)

@export
@register()
class TestColumnCopy(unittest.TestCase):
    def runTest(self):
        type_ = Types.BigInt()
        data = pygraphblas.Matrix.sparse(type_.as_pygrb, nrows=10, ncols=10)
        name = "Test"
        type_ = Types.BigInt()
        column = Column(name, data, type_=type_)
        column2 = column.copy()
        self.assertEqual(column2.name, name)
        self.assertEqual(column2.hidden, False)
        self.assertEqual(column2.data, data)
        self.assertIsNot(column2.data, column.data)
        self.assertIs(column2.gbtype, type_.as_pygrb)

@export
@register()
class TestColumnScalarInitText(unittest.TestCase):
    def runTest(self):
        type_ = Types.Text()
        name = "Test"
        data = pygraphblas.Scalar.from_type(type_.as_pygrb)
        column = Column(name, data, type_=type_)
        self.assertEqual(column.name, name)
        self.assertEqual(column.hidden, False)
        self.assertIs(column.data, data)
        self.assertIs(column.gbtype, type_.as_pygrb)
        self.assertTrue(column.is_scalar)
        self.assertFalse(column.is_vector)
        self.assertFalse(column.is_matrix)
        self.assertFalse(column.is_boolean)
        self.assertEqual(column.nptype, type_.as_numpy)
        self.assertEqual(ndim(column.data), 0)
