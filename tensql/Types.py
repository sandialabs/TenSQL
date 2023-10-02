import abc
import re
import ast

from typing import Set

from pygraphblas.base import lib as pygrb_lib
from pygraphblas.base import ffi as pygrb_ffi
from pygraphblas import unary_op, binary_op
import h5py
import numba
import numpy as np
import pygraphblas.types
import numba
from ._PyHashOCNT import npbytes2npstr, npstr2npbytes

from . import PyObjectHelper

type_registry = {}

def register_type(cls):
  type_registry[cls.__name__.upper()] = cls
  return cls

class Type:
  _as_pygrb = None
  _as_numpy = None
  _as_python = None
  _as_h5py = None
  _unary_ops = set()
  _binary_ops = set()
  _sideload = False
  _is_string = False

  @property
  def name(self) -> str:
    return type(self).__name__ + "()"

  @classmethod
  @property
  def as_pygrb(cls):
    return cls._as_pygrb

  @classmethod
  @property
  def as_numpy(cls):
    return cls._as_numpy

  @classmethod
  @property
  def as_python(cls):
    return cls._as_python

  @classmethod
  @property
  def as_h5py(cls):
    return cls._as_h5py

  @classmethod
  @property
  def unary_ops(cls) -> Set[str]:
    return cls._unary_ops

  @classmethod
  @property
  def binary_ops(cls) -> Set[str]:
    return cls._binary_ops

  @classmethod
  @property
  def sideload(cls) -> bool:
    return cls._sideload

  @classmethod
  @property
  def is_string(cls) -> bool:
    return cls._is_string

#def boolean_binary_op(arg_type, nopython=True):
#    """ Based on code from PyGraphBLAS """
#    def inner(func):
#        func_name = func.__name__
#        sig = numba.void(
#            numba.types.CPointer(pygraphblas.types.INT8._numba_t),
#            numba.types.CPointer(arg_type._numba_t),
#            numba.types.CPointer(arg_type._numba_t),
#        )
#        inner_sig = numba.boolean(arg_type._numba_t, arg_type._numba_t)
#        jitfunc = numba.jit(inner_sig, nopython=nopython, forceobj=not nopython)(func)
#
#        @numba.cfunc(sig, nopython=True)
#        def wrapper(z, x, y):  # pragma: no cover
#            result = False
#            result = jitfunc(x[0], y[0])
#            z[0] = result
#
#        out = pygrb_ffi.new("GrB_BinaryOp*")
#        pygrb_lib.GrB_BinaryOp_new(
#            out,
#            pygrb_ffi.cast("GxB_binary_function", wrapper.address),
#            pygraphblas.types.BOOL._gb_type,
#            arg_type._gb_type,
#            arg_type._gb_type,
#        )
#
#        return pygraphblas.binaryop.BinaryOp(func_name, arg_type.__name__, out[0])
#
#    return inner

@register_type
class Boolean(Type):
  _as_pygrb = pygraphblas.types.BOOL
  _as_numpy = np.bool_
  _as_h5py = np.bool_
  _as_python = bool

  _unary_ops = {"NOT", "ONE"}
  _binary_ops = {"==", "!=", "AND", "OR"}

class NumericType(Type):
  _unary_ops = {"-", "NOT", "ABS", "ONE"}
  _binary_ops = {"+", "-", "*", "/", "==", "!=", "<", ">", "<=", ">=", "AND", "OR"}

@register_type
class TinyInt(NumericType):
  _as_pygrb = pygraphblas.types.INT8
  _as_numpy = np.int8
  _as_h5py = np.int8
  _as_python = int

@register_type
class SmallInt(NumericType):
  _as_pygrb = pygraphblas.types.INT16
  _as_numpy = np.int16
  _as_h5py = np.int16
  _as_python = int

@register_type
class Integer(NumericType):
  _as_pygrb = pygraphblas.types.INT32
  _as_numpy = np.int32
  _as_h5py = np.int32
  _as_python = int

@register_type
class BigInt(NumericType):
  _as_pygrb = pygraphblas.types.INT64
  _as_numpy = np.int64
  _as_h5py = np.int64
  _as_python = int

@register_type
class Real(NumericType):
  _as_pygrb = pygraphblas.types.FP32
  _as_numpy = np.float32
  _as_h5py = np.float32
  _as_python = float

@register_type
class DoublePrecision(NumericType):
  _as_pygrb = pygraphblas.types.FP64
  _as_numpy = np.float64
  _as_h5py = np.float64
  _as_python = float

class SideLoadType(Type):
  pass

class StringType(SideLoadType):
  _binary_ops = {"==", "!="}
  _sideload = True
  _is_string = True

  def serialize(self, value):
    if isinstance(value, str):
      return value.encode('utf-8')
    elif isinstance(value, np.ndarray):
      if value.size > 0:
        ret = np.empty_like(value)
        npstr2npbytes(value, ret)
      else:
        ret = np.empty_like(value, dtype=object)
      return ret
    else:
      raise ValueError(f"Unsupported type: {type(value)}")

#  def serialize_bulk(self, value_array: np.ndarray):
#    

  def deserialize(self, b_value):
    if isinstance(b_value, bytes):
      return b_value.decode('utf-8')
    elif isinstance(b_value, h5py.Dataset):
      return np.array(b_value.asstr())
    elif isinstance(b_value, np.ndarray):
      if b_value.size > 0:
        ret = np.empty_like(b_value)
        npbytes2npstr(b_value, ret)
        #decoder = np.vectorize(lambda x: x.decode('UTF-8'))
        #ret = decoder(b_value).astype(object)
      else:
        ret = np.zeros_like(b_value, dtype=object)
      return ret
    else:
      raise ValueError(f"Unsupported type: {type(value)}")

class StringGrbTypeBase(PyObjectHelper.PyObject):
#  default_one = 1
#  default_zero = 0
#  _gb_type = pygrb_lib.GrB_INT64
#  _c_type = "int64_t"
#  _typecode = "q"
#  _numba_t = numba.int64
#  _numpy_t = np.int64
#  _base_name = "INT64"
  _max_length = None

  @pygraphblas.unary_op(pygraphblas.INT64)
  def ONE(x):
    return 1

  @pygraphblas.unary_op(pygraphblas.INT64)
  def ZERO(x):
    return 0

  @pygraphblas.unary_op(pygraphblas.INT64)
  def IDENTITY(x):
    return x

  @pygraphblas.binary_op(pygraphblas.INT64)
  def FIRST(x, y):
    return x

  @pygraphblas.binary_op(pygraphblas.INT64)
  def SECOND(x, y):
    return y

  @pygraphblas.binary_op(pygraphblas.INT64)
  def ANY(x, y):
    return x

#  @classmethod
#  def _from_value(cls, value):
#    return pygraphblas.INT64._from_value(value)
#
#  @classmethod
#  def _to_value(cls, value):
#    return pygraphblas.INT64._to_value(value)

#MIN,MAX,PLUS,FIRSTI,FIRSTI1,FIRSTJ,FIRSTJ1,SECONDI,SECONDI1,SECONDJ,SECONDJ1,PAIR,LOR,LAND,LXOR

class CharGrbTypeBase(StringGrbTypeBase):
  pass

@register_type
class Char(StringType):
  _as_h5py = h5py.string_dtype('utf-8')
  _as_python = str
  _sideload = False

  def __init__(self, max_length: int):
    assert isinstance(max_length, int)
    assert max_length > 0
    self._max_length = max_length

    class CharGrbType(CharGrbTypeBase):
      _max_length = self._max_length
    
    self._as_pygrb = CharGrbType

  @property
  def name(self) -> str:
    return f"{type(self).__name__}({self._max_length})"

  @property
  def max_length(self) -> int:
    return self._max_length

  @property
  def as_numpy(self):
    return np.dtype(f'U{self.max_length}')

  @property
  def as_h5py(self):
    return h5py.string_dtype('utf-8', self.max_length * 4)

  @property
  def as_pygrb(self):
    return self._as_pygrb


class VarCharGrbTypeBase(StringGrbTypeBase):
  pass

@register_type
class VarChar(StringType):
  _as_numpy = np.object_
  _as_h5py = h5py.string_dtype('utf-8')
  _as_python = str

  def __init__(self, max_length: int):
    assert isinstance(max_length, int)
    assert max_length > 0
    self._max_length = max_length

  @property
  def name(self) -> str:
    return f"{type(self).__name__}({self._max_length})"

  @property
  def max_length(self) -> int:
    return self._max_length

  @property
  def as_pygrb(self):
    class VarCharGrbType(VarCharGrbTypeBase):
      _max_length = self._max_length

    return VarCharGrbType

class TextGrbType(StringGrbTypeBase):
  pass

@register_type
class Text(StringType):
  _as_numpy = np.object_
  _as_h5py = h5py.string_dtype('utf-8')
  _as_python = str

  @property
  def as_pygrb(self):
    return TextGrbType

def from_pygrb(grb_type):
  if (type(grb_type) == pygraphblas.types.BOOL) or (grb_type is pygraphblas.types.BOOL):
    return Boolean()
  elif (type(grb_type) == pygraphblas.types.INT8) or (grb_type is pygraphblas.types.INT8):
    return TinyInt()
  elif (type(grb_type) == pygraphblas.types.INT16) or (grb_type is pygraphblas.types.INT16):
    return SmallInt()
  elif (type(grb_type) == pygraphblas.types.INT32) or (grb_type is pygraphblas.types.INT32):
    return Integer()
  elif (type(grb_type) == pygraphblas.types.INT64) or (grb_type is pygraphblas.types.INT64):
    return BigInt()
  elif (type(grb_type) == pygraphblas.types.FP32) or (grb_type is pygraphblas.types.FP32):
    return Real()
  elif (type(grb_type) == pygraphblas.types.FP64) or (grb_type is pygraphblas.types.FP64):
    return DoublePrecision()
  elif isinstance(grb_type, CharGrbTypeBase) or issubclass(grb_type, CharGrbTypeBase):
    return Char(grb_type.max_length)
  elif isinstance(grb_type, VarCharGrbTypeBase) or issubclass(grb_type, VarCharGrbTypeBase):
    return VarChar(grb_type.max_length)
  elif isinstance(grb_type, TextGrbType) or issubclass(grb_type, TextGrbType):
    return Text()
  else:
    raise NotImplementedError("Invalid grb_type")

def from_numpy(np_type):
  if isinstance(grb_type, np.bool_) or issubclass(grb_type, np.bool_):
    return Boolean()
  elif isinstance(grb_type, np.int8) or issubclass(grb_type, np.int8):
    return TinyInt()
  elif isinstance(grb_type, np.int16) or issubclass(grb_type, np.int16):
    return SmallInt()
  elif isinstance(grb_type, np.int32) or issubclass(grb_type, np.int32):
    return Integer()
  elif isinstance(grb_type, np.int64) or issubclass(grb_type, np.int64):
    return BigInt()
  elif isinstance(grb_type, np.float32) or issubclass(grb_type, np.float32):
    return Real()
  elif isinstance(grb_type, np.float64) or issubclass(grb_type, np.float64):
    return DoublePrecision()
  else:
    raise NotImplementedError(f"Unsupported np_type: {np_type!s}")

def from_native(py_type):
  if issubclass(py_type, int):
    return BigInt()
  elif issubclass(py_type, float):
    return DoublePrecision()
  elif issubclass(py_type, str):
    return Text()
  else:
    raise NotImplementedError(f"Unsupported py_type: {py_type!s}")

def from_string(type_decl: str):
  re_typename = r'(\w+)(\(.*\))'
  match = re.fullmatch(re_typename, type_decl)
  if match is None:
    raise ValueError(f"Invalid type string: {type_decl!r}")
  typename = match.group(1).upper()
  args = ast.literal_eval(match.group(2))
  if not isinstance(args, tuple):
    args = (args,)
  if typename not in type_registry:
    raise LookupError(f"Invalid type name: {typename!r}")
  return type_registry[typename](*args)
