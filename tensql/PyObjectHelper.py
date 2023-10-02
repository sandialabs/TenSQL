import numpy as np
import numba
import pygraphblas.types
import pygraphblas.binaryop
from pygraphblas.base import lib as pygrb_lib, ffi as pygrb_ffi

from ._PyObjectHelper import lib as pyoh_lib, ffi as pyoh_ffi

class AddressWrapper:
  def __init__(self, name):
    self.address = pyoh_ffi.addressof(pyoh_lib, name)

class PyObject(pygraphblas.types.INT64):
#  default_one = 1
#  default_zero = 0
#  _gb_type = pygrb_lib.GrB_INT64
#  _c_type = "void*"
#  _typecode = "q"
#  _numba_t = numba.int64
#  _numpy_t = np.int64
#  _base_name = "UDT"
  pass

def load_helper_function(op, *, boolean):
  return pygraphblas.binaryop.BinaryOp(
    op, "PyObject", 
    AddressWrapper(f"PyObject_{op}"), 
    udt=PyObject,
    boolean=boolean
  )

PyObject.EQ = load_helper_function("EQ", boolean=True)
PyObject.NE = load_helper_function("NE", boolean=True)
PyObject.LT = load_helper_function("LT", boolean=True)
PyObject.GT = load_helper_function("GT", boolean=True)
PyObject.LE = load_helper_function("LE", boolean=True)
PyObject.GE = load_helper_function("GE", boolean=True)

PyObject.ANY = load_helper_function("ANY", boolean=False)
PyObject.FIRST = load_helper_function("FIRST", boolean=False)
PyObject.SECOND = load_helper_function("SECOND", boolean=False)
