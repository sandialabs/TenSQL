from .._PyHashOCNT import PyHashOCNT
import unittest
from ..util import export
from .test_util import register, TmpDirMixin
import sys

__all__ = []

@export
@register()
class TestOCNT_init(unittest.TestCase):
    def runTest(self):
        ocnt = PyHashOCNT(16)
        self.assertEqual(ocnt.size, 0)
        del ocnt

@export
@register()
class TestOCNT_insert_delete(unittest.TestCase):
    def runTest(self):
        x = "asdfeasdfebcxcqse"
        x_refcount = sys.getrefcount(x)
        ocnt = PyHashOCNT(16)
        ocnt.insert(x)
        self.assertEqual(ocnt.size, 1)
        self.assertEqual(x_refcount + 1, sys.getrefcount(x))
        del ocnt
        self.assertEqual(x_refcount, sys.getrefcount(x))

@export
@register()
class TestOCNT_insert_remove_delete(unittest.TestCase):
    def runTest(self):
        x = "asdfeasdfeewafeasef"
        x_refcount = sys.getrefcount(x)
        ocnt = PyHashOCNT(16)
        self.assertEqual(ocnt.size, 0)

        ocnt.insert(x)
        self.assertEqual(ocnt.size, 1)
        self.assertEqual(x_refcount + 1, sys.getrefcount(x))

        ocnt.remove(x)
        self.assertEqual(x_refcount, sys.getrefcount(x))

        del ocnt
        self.assertEqual(x_refcount, sys.getrefcount(x))

@export
@register()
class TestOCNT_insert_decode_remove_delete(unittest.TestCase):
    def runTest(self):
        x = "asdfeasdfesdfesdf"
        x_refcount = sys.getrefcount(x)
        ocnt = PyHashOCNT(16)

        ocnt.insert(x)
        self.assertEqual(ocnt.size, 1)
        self.assertEqual(x_refcount + 1, sys.getrefcount(x))

        x_id = id(x)
        self.assertEqual(x_refcount + 1, sys.getrefcount(x))

        y = ocnt.decode(x_id)
        self.assertEqual(x_refcount+2, sys.getrefcount(x))

        del y
        self.assertEqual(x_refcount+1, sys.getrefcount(x))

        ocnt.remove(x)
        self.assertEqual(x_refcount, sys.getrefcount(x))

        del ocnt
        self.assertEqual(x_refcount, sys.getrefcount(x))
