# distutils: language = c++
# sources = siphash.c

from typing import Optional, Sequence, Any, Tuple, List

from libcpp cimport bool
from libc.stdint cimport *
from libc.stdlib cimport malloc, free

import pygraphblas.types
import contextlib
import sys

import numpy as np
cimport numpy as np

import cython

cdef extern from "stdio.h":
  int write(int handle, const char* buffer, unsigned int count)
  ssize_t read(int fd, void *buf, size_t count)

cdef extern from "<algorithm>" namespace "std" nogil:
  void swap[T](T& a, T& b) except +

cdef extern from "HashStore.hpp":
  cdef cppclass HashStore[KEY_T, VALUE_T]:
    HashStore(size_t size)
    HashStore(const HashStore[KEY_T, VALUE_T]& copy)

    @staticmethod
    HashStore[KEY_T, VALUE_T]* copy(const HashStore[KEY_T, VALUE_T]& other)

    bool put(const KEY_T &key, const VALUE_T &value)
    VALUE_T* get(const KEY_T &key)
    VALUE_T* setdefault(const KEY_T &key, const VALUE_T & value)
    bool remove(const KEY_T &key)
    size_t getSize() const
    size_t toArrays(KEY_T** keys, VALUE_T** values, size_t max_entries)

cdef extern from "PythonObj.hpp":
  cdef cppclass PythonObj nogil:
    PythonObj()
    PythonObj(object)
    void set(object)
    object toPython(bool incref)
    ssize_t toInteger()
    ssize_t getRefCount()

cdef extern from "Python.h":
  cdef int PyBytes_Check(object)
  cdef object PyBytes_FromStringAndSize(const char *, ssize_t)
  cdef int PyBytes_AsStringAndSize(object, char**, ssize_t*)

  cdef int PyUnicode_Check(object)
  cdef object PyUnicode_FromStringAndSize(const char *, ssize_t)
  cdef char* PyUnicode_AsUTF8AndSize(object, ssize_t*)
  cdef object PyUnicode_DecodeUTF8(const char*, ssize_t, const char*)

@cython.boundscheck(False)
@cython.wraparound(False)
def npbytes2npstr(np.ndarray[object, ndim=1] array_in, np.ndarray[object, ndim=1] array_out):
  assert array_in.shape[0] == array_out.shape[0]
  cdef int64_t N = array_in.shape[0]
  cdef ssize_t vlen = 0
  cdef char* buf = NULL
  cdef object bytes_obj = None

  for it in range(N):
    bytes_obj = array_in[it]
    if PyBytes_Check(bytes_obj):
      PyBytes_AsStringAndSize(bytes_obj, &buf, &vlen)
      array_out[it] = PyUnicode_FromStringAndSize(buf, vlen)
    else:
      raise ValueError("Object is not a bytes instance")

@cython.boundscheck(False)
@cython.wraparound(False)
def npstr2npbytes(np.ndarray[object, ndim=1] array_in, np.ndarray[object, ndim=1] array_out):
  assert array_in.shape[0] == array_out.shape[0]
  cdef int64_t N = array_in.shape[0]
  cdef ssize_t vlen = 0
  cdef const char* buf = NULL
  cdef object str_obj = None

  for it in range(N):
    str_obj = array_in[it]
    if PyUnicode_Check(str_obj):
      buf = PyUnicode_AsUTF8AndSize(str_obj, &vlen)
      array_out[it] = PyBytes_FromStringAndSize(buf, vlen)
    else:
      array_out[it] = None
      raise ValueError("Object is not a string instance")

@cython.boundscheck(False)
@cython.wraparound(False)
def bulk_serialize_strings(np.ndarray[object, ndim=1] array_in, np.ndarray[np.int64_t, ndim=1] array_out, int fout_fd):
  assert array_in.shape[0] == array_out.shape[0]
  cdef int64_t N = array_in.shape[0]
  cdef ssize_t offset = 0
  cdef ssize_t vlen = 0
  cdef const char* buf = NULL
  cdef object str_obj = None

  for it in range(N):
    str_obj = array_in[it]
    if PyUnicode_Check(str_obj):
      buf = PyUnicode_AsUTF8AndSize(str_obj, &vlen)
      if write(fout_fd, buf, vlen) != vlen:
        raise IOError("Input/Output Error")
      offset += vlen
      array_out[it] = offset
    else:
      raise ValueError("Object is not a string")

@cython.boundscheck(False)
@cython.wraparound(False)
def bulk_deserialize_strings(np.ndarray[object, ndim=1] array_out, np.ndarray[np.int64_t, ndim=1] array_in, int fin_fd):
  assert array_in.shape[0] == array_out.shape[0]
  cdef int64_t N = array_in.shape[0]
  cdef ssize_t prev_offset = 0
  cdef ssize_t next_offset = 0
  cdef ssize_t vlen = 0
  cdef const char* buf = NULL
  cdef object str_obj = None

  cdef ssize_t buff_size = 64 * 1024
  cdef char* buff = <char*>malloc(buff_size)
  cdef object empty_string = ""

  for it in range(N):
    next_offset = array_in[it]
    vlen = next_offset - prev_offset
    if vlen > 0:
      if vlen > buff_size:
        free(buff)
        buff = <char*>malloc(vlen)
        if buff == NULL:
          raise MemoryError("Failed to allocate buffer")

      if read(fin_fd, buff, vlen) != vlen:
        raise IOError("Input/Output Error")

      str_obj = PyUnicode_DecodeUTF8(buff, vlen, "strict")
      
      if <void*>str_obj == NULL:
        raise UnicodeError("Failed to decode utf-8 string")
    else:
      str_obj = empty_string

    array_out[it] = empty_string
      
    prev_offset = next_offset

  free(buff)

cdef class PyHashOCNT:
  cdef HashStore[PythonObj, int64_t]* _counts

  def __cinit__(self, size: int):
    cdef size_t c_size = size
    self._counts = new HashStore[PythonObj, int64_t](c_size)
    #print("Creating PyHashOCNT", id(self))

  def copy(self) -> PyHashOCNT:
    cdef int64_t orig_size = self._counts[0].getSize()

    #print("Creating empty PyHashOCNT") ; sys.stdout.flush()
    ret = PyHashOCNT(0)
    #print("Deleting empty PyHashOCNT HashStore") ; sys.stdout.flush()
    del ret._counts
    #print("Creating new PyHashOCNT HashStore") ; sys.stdout.flush()
    ret._counts = HashStore[PythonObj, int64_t].copy(self._counts[0])

    #print("copy()", "Original size:", orig_size, "New size:", ret._counts[0].getSize())
    return ret

  @staticmethod
  def decode(idx: int) -> Any:
    cdef PythonObj c_key
    cdef intptr_t c_idx = idx
    c_key.set(<object><void*>c_idx)
    return c_key.toPython(True)

  @staticmethod
  def decode_many(np.ndarray[np.int64_t, ndim=1] arr) -> np.ndarray[object]:
    cdef int64_t N = arr.size
    cdef PythonObj c_val
    cdef intptr_t c_idx
    cdef np.ndarray[object, ndim=1] ret = np.empty([N], dtype=object)
    cdef int64_t it

    for it in range(N):
        c_idx = arr[it]
        c_val.set(<object><void*>c_idx)
        ret[it] = c_val.toPython(True)

    return ret

  def insert_many(self, seq: Sequence[str]) -> List[int]:
    ret = []
    for item in seq:
      #print(item)
      ret.append(self.insert(item)[0])
    return ret

  def insert_many_encoded_numpy(self, np.ndarray[int64_t, ndim=1] arr) -> np.ndarray:
    cdef int64_t N = arr.size
    cdef int64_t it = 0
    cdef PythonObj c_key
    cdef int64_t* opt_count = NULL
    cdef ssize_t ref_count = 0

    for it in range(arr.size):
        c_key.set(<object><void*>arr[it])
        ref_count = c_key.getRefCount()
        opt_count = self._counts.get(c_key)
        if opt_count != NULL:
            opt_count[0] += 1
        else:
            self._counts.put(c_key, 1)

  def insert_many_numpy(self, np.ndarray[object, ndim=1] arr) -> np.ndarray:
    cdef int64_t N = arr.size
    cdef np.ndarray ret = np.zeros([N], dtype=np.int64)
    cdef int64_t it = 0
    cdef PythonObj c_key
    cdef int64_t* opt_count = NULL

    for it in range(arr.size):
        c_key.set(arr[it])
        opt_count = self._counts.get(c_key)
        ret[it] = c_key.toInteger()
        if opt_count != NULL:
            opt_count[0] += 1
        else:
            self._counts.put(c_key, 1)

    return ret

  def insert(self, key: Any, *, increment=1) -> Tuple[int, int]:
    cdef PythonObj c_key
    cdef int64_t* opt_count = NULL

    c_key.set(key)
    opt_count = self._counts.get(c_key)

    if opt_count != NULL:
      opt_count[0] += increment
      return c_key.toInteger(), opt_count[0]
    else:
      self._counts.put(c_key, increment)
      return c_key.toInteger(), increment

  def _print_debug(self, verbose=True) -> None:
    cdef size_t alloc_size = self._counts.getSize()
    cdef PythonObj** keys = <PythonObj**>malloc(sizeof(PythonObj*) * alloc_size)
    cdef int64_t** values = <int64_t**>malloc(sizeof(int64_t*) * alloc_size)
    cdef size_t num_copied = self._counts.toArrays(keys, values, alloc_size)
    cdef PythonObj* c_key = NULL
    cdef ssize_t ref_count = 0

    print("PyHashOCNT size", alloc_size)
    sys.stdout.flush()

    for i in range(num_copied):
      c_key = keys[i]
      ref_count = c_key[0].getRefCount()
      
      if verbose:
        print("IntPtr", c_key[0].toInteger(), "Object", c_key[0].toPython(True), "PyRefCount:", ref_count, "OCNT:", values[i][0])
      else:
        print("IntPtr", c_key[0].toInteger(), "PyRefCount:", ref_count, "OCNT:", values[i][0])
      sys.stdout.flush()

    free(keys)
    free(values)


  def remove(self, key: Any, *, decrement=1) -> Optional[int]:
    cdef PythonObj c_key
    cdef int64_t* opt_count = NULL
    cdef int64_t new_count = 0

    c_key.set(key)
    opt_count = self._counts.get(c_key)

    if opt_count != NULL:
      opt_count[0] -= decrement
      new_count = opt_count[0]

      if opt_count[0] <= 0:
        self._counts.remove(c_key)

      return new_count
    else:
      return None

  @contextlib.contextmanager
  def checkpoint(self):
    cdef PyHashOCNT backup = self.copy()

    try:
      yield self
    except:
      swap(self._counts, backup._counts)
      raise

  @property
  def size(self) -> int:
    return self._counts.getSize()

  def __dealloc__(self):
    #print("Deallocating", hex(int(<int64_t>self._counts)))
    #print("Destroying PyHashOCNT", id(self))
    del self._counts
    #print("Finished Deallocating", hex(int(<int64_t>self._counts)))
