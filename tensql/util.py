import abc
import hashlib
import collections
import inspect
import threading
import json
from tkinter import N

from typing import List, Optional, Union, Any, Type, Sequence, Tuple

import numpy as np
from pygraphblas.types import Type as GBType
from pygraphblas import Matrix, Vector, Scalar
import pygraphblas
import pygraphblas.types
from pygraphblas.base import NULL, ffi

from . import Types

class Panic(pygraphblas.base.GraphBLASException):
  pass

class OutOfMemory(pygraphblas.base.GraphBLASException):
  pass

class InsufficientSpace(pygraphblas.base.GraphBLASException):
  pass

class InvalidObject(pygraphblas.base.GraphBLASException):
  pass

class IndexOutOfBounds(pygraphblas.base.GraphBLASException):
  pass

class EmptyObject(pygraphblas.base.GraphBLASException):
  pass

class UninitializedObject(pygraphblas.base.GraphBLASException):
  pass

class NullPointer(pygraphblas.base.GraphBLASException):
  pass

class InvalidValue(pygraphblas.base.GraphBLASException):
  pass

class InvalidIndex(pygraphblas.base.GraphBLASException):
  pass

class DomainMismatch(pygraphblas.base.GraphBLASException):
  pass

class DimensionMismatch(pygraphblas.base.GraphBLASException):
  pass

class OutputNotEmpty(pygraphblas.base.GraphBLASException):
  pass

class GrbNotImplemented(pygraphblas.base.GraphBLASException):
  pass

pygraphblas.base._error_codes.update({
  -1: UninitializedObject,
  -2: NullPointer,
  -3: InvalidValue,
  -4: InvalidIndex,
  -5: DomainMismatch,
  -6: DimensionMismatch,
  -7: OutputNotEmpty,
  -8: GrbNotImplemented,
  -101: Panic,
  -102: OutOfMemory,
  -103: InsufficientSpace,
  -104: InvalidObject,
  -105: IndexOutOfBounds,
  -106: EmptyObject
})

class DummyTensor:
    def __init__(self, shape: Tuple[int, ...]):
        self._shape = shape

    @property
    def shape(self) -> int:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)


GBTensor = Union[Matrix, Vector, Scalar, DummyTensor]
Axes = Sequence[int]

UnaryPyOps = {
    "-": lambda x: -x,
    "NOT": lambda x: not x,
    "ABS": lambda x: abs(x),
    "ONE": lambda x: 1,
}

UnaryGrbOps = {
    "-": "AINV",
    "NOT": "LNOT",
    "ABS": "ABS",
    "ONE": "ONE"
}

BinaryArithmeticPyOps = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "//": lambda x, y: x // y,
    "%": lambda x, y: x % y,
}

BinaryArithmeticGrbOps = {
    "+": "PLUS",
    "-": "MINUS",
    "*": "TIMES",
    "/": "DIV",
}

BinaryBooleanPyOps = {
    "==": lambda x, y: x == y,
    "!=": lambda x, y: x != y,
    ">": lambda x, y: x > y,
    "<": lambda x, y: x < y,
    "<=": lambda x, y: x <= y,
    ">=": lambda x, y: x >= y,
    "AND": lambda x, y: x and y,
    "OR": lambda x, y: x or y,
}

BinaryBooleanGrbOps = {
    "==": "EQ",
    "!=": "NE",
    ">": "GT",
    "<": "LT",
    "<=": "GE",
    ">=": "LE",
    "AND": "TIMES",
    "OR": "PLUS",
}


def grb_shape(obj: GBTensor) -> Sequence[int]:
    if isinstance(obj, Scalar):
        return tuple()
    elif isinstance(obj, Vector):
        return (obj.size,)
    elif isinstance(obj, Matrix):
        return (obj.nrows, obj.ncols)
    elif isinstance(obj, DummyTensor):
        return obj.shape
    else:
        raise ValueError(
            f"Prototype is only implemented for matrices, vectors, and scalars, not {type(obj)=}"
        )


shape = grb_shape


def ndim(obj: GBTensor) -> int:
    if isinstance(obj, Scalar):
        return 0
    elif isinstance(obj, Vector):
        return 1
    elif isinstance(obj, Matrix):
        return 2
    elif isinstance(obj, DummyTensor):
        return obj.ndim
    else:
        raise ValueError(
            f"Prototype is only implemented for matrices, vectors, and scalars, not {type(obj)=}"
        )

def scalar_add(first: Scalar, second: Scalar, *, op: Optional[str]=None, inplace: bool=False):
    if inplace:
        data = first
    else:
        data = Scalar.from_type(first.type)

    if op is None:
        op = "+"

    if first.nvals == 0 and second.nvals == 0:
        pass
    elif first.nvals > 0 and second.nvals == 0:
        data[None] = first[None]
    elif first.nvals == 0 and second.nvals > 0:
        data[None] = second[None]
    elif first.nvals > 0 and second.nvals > 0:
        data[None] = BinaryArithmeticPyOps[op](
            first[None], second[None]
        )

    return data


class ABCIdGenerator(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, value):
        pass


class SequentialIdGenerator(ABCIdGenerator):
    def __init__(self):
        self.counter = 0
        self.lock = threading.Lock()

    def increment(self, amount=1):
        with self.lock:
            ret = self.counter
            self.counter += amount
        return ret

    def __call__(self, value):
        return self.increment()

class HashIdGenerator(ABCIdGenerator):
    KEYSPACE_NIBBLES = 15  # 60 bits of hash

    def __init__(self, algo="sha256", salt=None):
        self.hasher = hashlib.new(algo)
        if salt is not None:
            self.hasher.update(salt)

    def __call__(self, value):
        hasher = self.hasher.copy()
        if isinstance(value, str):
            hasher.update(value.encode("utf-8"))
        else:
            hasher.update(value)
        return int(hasher.hexdigest()[: self.KEYSPACE_NIBBLES], 16)


BinaryPyOps = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "//": lambda x, y: x // y,
    "%": lambda x, y: x % y,
    "==": lambda x, y: x == y,
    "!=": lambda x, y: x != y,
    ">": lambda x, y: x > y,
    "<": lambda x, y: x < y,
    "<=": lambda x, y: x <= y,
    ">=": lambda x, y: x >= y,
    "AND": lambda x, y: x and y,
    "OR": lambda x, y: x or y,
    "FIRST": lambda x, y: x,
    "SECOND": lambda x, y: y,
}

BinaryGrbOps = {
    "+": "PLUS",
    "-": "MINUS",
    "*": "TIMES",
    "/": "DIV",
    "==": "EQ",
    "!=": "NE",
    ">": "GT",
    "<": "LT",
    ">=": "GE",
    "<=": "LE",
    "AND": "TIMES",
    "OR": "PLUS",
    "FIRST": "FIRST",
    "SECOND": "SECOND",
    "LAND": "LAND",
    "LOR": "LOR",
}


class EncoderDecoder:
    def __init__(self, id_generator: SequentialIdGenerator):
        self.id_generator = id_generator
        self.encoder = collections.OrderedDict()
        self.decoder = {}
        self.lock = threading.Lock()
        self._np_encode = np.vectorize(lambda x: self[x], otypes=[np.uint64])

    def __getitem__(self, name: str) -> Any:
        return self.get(name)[0]

    def get(self, name: str) -> Any:
        ret = self.encoder.get(name)
        if ret is None:
            with self.lock:
                ret = self.encoder.get(name)
                if ret is None:
                    ret = self.id_generator(name)
                    self.encoder[name] = ret
                    self.decoder[ret] = name

            return ret, True
        else:
            return ret, False

    def __setitem__(self, name: str, value: Any) -> None:
        with self.lock:
            prev_value = self.encoder.get(name)
            if prev_value is None:
                self.encoder[name] = value
                self.decoder.setdefault(value, name)
            else:
                raise ValueError(f"Key {name} was already set")

    def __contains__(self, name: str) -> bool:
        return name in self.encoder

    def encode(self, name: str) -> Any:
        return self.encoder.get(name)

    def encode_many(self, names: Sequence[str]) -> np.ndarray:
        return self._np_encode(names)

    def decode(self, ident: Any) -> str:
        return self.decoder.get(ident)

    def items(self):
        for k, v in self.encoder.items():
            yield k, v

    def __len__(self):
        return len(self.encoder)

    def to_json(self):
        return json.dumps(self.encoder)


def export(obj):
    """
    A helpful decorator used to control __all__
    Args:
        obj: the object whose name should be added to __all__
    Returns:
        obj
    """
    inspect.getmodule(obj).__all__.append(obj.__name__)
    return obj

def vector_grb_check_success(vector, res):
    if res != pygraphblas.lib.GrB_SUCCESS:
        error_string = ffi.new("char**")
        pygraphblas.lib.GrB_Vector_error(error_string, vector._vector[0])
        raise pygraphblas.base._error_codes[res](ffi.string(error_string[0]))

def matrix_grb_check_success(vector, res):
    if res != pygraphblas.lib.GrB_SUCCESS:
        error_string = ffi.new("char**")
        pygraphblas.lib.GrB_Matrix_error(error_string, vector._vector[0])
        raise pygraphblas.base._error_codes[res](ffi.string(error_string[0]))

def grb_check_success(res):
    pygraphblas.base._check(res)

def extract_indices(obj, axis, typ=None):
    if typ is None:
        typ = Types.BigInt()

    if not isinstance(typ, Types.BigInt):
        raise NotImplementedError("Primary key types other than BigInt are not yet supported.")
    
    assert isinstance(typ, Types.BigInt)

    if isinstance(obj, Scalar):
        raise ValueError("Scalar has no indices")
    elif isinstance(obj, Vector):
        if axis != 0:
          raise ValueError("axis must be 0 when obj argument is a vector")
        ret = Vector.sparse(pygraphblas.types.INT64, size=obj.size)

        nvals = obj.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new("GrB_Index[%s]" % nvals)
        X = NULL
        vector_grb_check_success(
          obj,
          obj.type._Vector_extractTuples(I, X, _nvals, obj._vector[0])
        )

        X = ffi.cast("int64_t[%s]" % nvals, I)

        grb_check_success(
          pygraphblas.lib.GrB_Vector_build_INT64(
            ret._vector[0],
            I, X,
            nvals,
            typ.as_pygrb.FIRST.get_op()
          )
        )
        return ret
    elif isinstance(obj, Matrix):
        if axis not in (0,1):
          raise ValueError("axis must be 0 or 1 when obj argument is a matrix")
        ret = Matrix.sparse(pygraphblas.types.INT64, nrows=obj.nrows, ncols=obj.ncols)

        nvals = obj.nvals
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new("GrB_Index[%s]" % nvals)
        J = ffi.new("GrB_Index[%s]" % nvals)
        X = NULL
        matrix_grb_check_success(
          obj,
          obj.type._Matrix_extractTuples(I, J, X, _nvals, obj._matrix[0])
        )

        if axis == 0:
          X = I
        elif axis == 1:
          X = J

        X = ffi.cast("int64_t[%s]" % nvals, X)

        grb_check_success(
          pygraphblas.lib.GrB_Matrix_build_INT64(
            ret._matrix[0],
            I, J, X,
            nvals,
            typ.as_pygrb.FIRST.get_op()
          )
        )
        return ret
    else:
        raise ValueError(
            "Prototype is only implemented for matrices, vectors, and scalars"
        )

def build_tensor(gbtype: pygraphblas.types.Type, shape: Sequence[int]) -> GBTensor:
    assert issubclass(gbtype, pygraphblas.types.Type)

    if len(shape) == 0:
        return Scalar.from_type(gbtype)
    elif len(shape) == 1:
        return Vector.sparse(gbtype, *shape)
    elif len(shape) == 2:
        return Matrix.sparse(gbtype, *shape)
    else:
        raise NotImplementedError(
            "Prototype only supports scalars, vectors, and matrices"
        )

def tensor_to_numpy(obj: GBTensor) -> Tuple[np.ndarray]:
    nvals, typ = obj.nvals, obj.type

    if ndim(obj) == 0:
        ret = np.zeros(shape=(obj.nvals,), dtype=typ._numpy_t)
        if obj.nvals > 0:
            ret[0] = obj[None]
        return (ret,)
    elif ndim(obj) == 1:
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new(f"GrB_Index[{nvals}]")
        X = ffi.new(f"{typ._c_type}[{nvals}]")
        vector_grb_check_success(
            obj,
            obj.type._Vector_extractTuples(I, X, _nvals, obj._vector[0])
        )
        return (
            np.frombuffer(ffi.buffer(I, ffi.sizeof(I)), dtype=np.uint64),
            np.frombuffer(ffi.buffer(X, ffi.sizeof(X)), dtype=typ._numpy_t)
        )
    elif ndim(obj) == 2:
        _nvals = ffi.new("GrB_Index[1]", [nvals])
        I = ffi.new(f"GrB_Index[{nvals}]")
        J = ffi.new(f"GrB_Index[{nvals}]")
        X = ffi.new(f"{typ._c_type}[{nvals}]")
        matrix_grb_check_success(
            obj,
            obj.type._Matrix_extractTuples(I, J, X, _nvals, obj._matrix[0])
        )
        return (
            np.frombuffer(ffi.buffer(I, ffi.sizeof(I)), dtype=np.uint64),
            np.frombuffer(ffi.buffer(J, ffi.sizeof(J)), dtype=np.uint64),
            np.frombuffer(ffi.buffer(X, ffi.sizeof(X)), dtype=typ._numpy_t)
        )
    else:
        raise NotImplementedError(
            "Prototype only supports scalars, vectors, and matrices"
        )

def fill_tensor_from_numpy(obj: GBTensor, *lists: Sequence[np.ndarray]) -> None:
    assert len(lists) == ndim(obj) + 1

    if ndim(obj) == 0:
        if lists[0].size > 0:
            obj[None] = lists[0][0]
        return
    elif ndim(obj) not in (1, 2):
        raise NotImplementedError(
            "Prototype only supports scalars, vectors, and matrices"
        )

    build_func_name_parts = ["GrB"]
    if ndim(obj) == 1:
        build_func_name_parts.append("Vector")
        grb_tensor = obj._vector[0]
    elif ndim(obj) == 2:
        build_func_name_parts.append("Matrix")
        grb_tensor = obj._matrix[0]

    build_func_name_parts.append("build")

    typ = obj.type
    build_func_name_parts.append(typ._base_name)

    build_func_name = "_".join(build_func_name_parts)
    assert hasattr(pygraphblas.lib, build_func_name)
    build_func = getattr(pygraphblas.lib, build_func_name)

    indices, value = lists[:-1], lists[-1]
    nvals = value.size
    assert all(idx.size == value.size for idx in indices)
    ffi_lists = [ffi.cast(f"uint64_t[{nvals}]", idx.ctypes.data) for idx in indices]
    ffi_lists.append(
        ffi.cast(f"{typ._c_type}[{nvals}]", value.ctypes.data)
    )

    #print([x for x in ffi_lists], nvals)

    grb_check_success(
        build_func(
            grb_tensor,
            *ffi_lists,
            nvals,
            typ.FIRST.get_op()
        )
    )
    

def initialize_tensor(
    gbtype: pygraphblas.types.Type, shape: Sequence[int], lists: Sequence[Sequence[Any]]
) -> GBTensor:
    try:
        is_gbtype_subclass = issubclass(gbtype, pygraphblas.types.Type)
    except TypeError:
        is_gbtype_subclass = False
    
    if not (isinstance(gbtype, pygraphblas.types.Type) or is_gbtype_subclass):
        raise ValueError(f"Expected instance of pygraphblas.types.Type, not {gbtype}")
    assert len(lists) == len(shape) + 1

    #lists = [np.array(l).tolist() for l in lists]

    if len(shape) == 0:
        ret = Scalar.from_type(gbtype)
        if len(lists[0]) == 0:
            pass
        elif len(lists[0]) == 1:
            ret[None] = lists[0][0]
        else:
            raise ValueError("Scalars should only have zero or one entries!")
        return ret
    elif len(shape) == 1:
        ret = Vector.sparse(gbtype, size=shape[0])
        if len(lists[0]) > 0:
            fill_tensor_from_numpy(
                ret,
                np.array(lists[0], dtype=np.uint64, copy=False),
                np.array(lists[1], dtype=gbtype._numpy_t, copy=False),
            )
    elif len(shape) == 2:
        ret = Matrix.sparse(
            gbtype,
            nrows=shape[0],
            ncols=shape[1]
        )
        if len(lists[0]) > 0:
            fill_tensor_from_numpy(
                ret,
                np.array(lists[0], dtype=np.uint64, copy=False),
                np.array(lists[1], dtype=np.uint64, copy=False),
                np.array(lists[2], dtype=gbtype._numpy_t, copy=False)
            )
#            ret = Matrix.from_lists(
#                I=lists[0],
#                J=lists[1],
#                V=lists[2],
#                nrows=shape[0],
#                ncols=shape[1],
#                typ=gbtype,
#            )
    else:
        raise ValueError("Prototype only supports scalars, vectors, and matrices")

    return ret
