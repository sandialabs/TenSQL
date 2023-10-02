"""
The Column class represents the data in a column of a table.
"""

from __future__ import annotations

import copy
import contextlib
from typing import Optional, List, Any, Tuple

from pygraphblas import Matrix, Vector, Scalar
import numpy as np

from .util import GBTensor, GBType, ndim, initialize_tensor, grb_shape, tensor_to_numpy
from . import Types
from ._PyHashOCNT import PyHashOCNT

class Column:
    def __init__(
        self,
        name: str,
        data: Optional[GBTensor],
        *,
        type_: Types.Type,
        hidden: bool = False,
        refcounts: Optional[PyHashOCNT] = None
    ) -> None:
        assert isinstance(name, str)
        assert isinstance(data, (Matrix, Vector, Scalar)) or data is None
        assert isinstance(type_, Types.Type)

        if data is not None:
          assert isinstance(type_.as_pygrb, data.type) or issubclass(type_.as_pygrb, data.type)

        self._type = type_
        self._name = name
        self._data = data
        self._hidden = hidden
        if self._type.sideload and refcounts is None:
            self._refcounts = PyHashOCNT(1024)
        elif self._type.sideload and refcounts is not None:
            self._refcounts = refcounts.copy()
        else:
            self._refcounts = None

    def __copy__(self):
        """
        Creates a copy of the column that points to a copy of the original data
        """
        if self._data is None:
            data = None
        else:
            data = self._data.dup()

        return Column(self._name, data, type_=self.type_, hidden=self.hidden, refcounts=self._refcounts)

    def copy(self):
        """
        Creates a copy of the column that points to a copy of the original data
        """
        return copy.copy(self)

    @property
    def name(self) -> str:
        """
        Returns the name of the column
        """
        return self._name

    @property
    def data(self) -> Optional[GBTensor]:
        """
        Returns the underlying graphblas data of the column
        """
        return self._data

    @property
    def values(self) -> np.ndarray:
        ret = tensor_to_numpy(self._data)[-1]
        if self.is_sideloaded:
            ret = self._refcounts.decode_many(ret)

        return ret

    @property
    def data_template(self) -> Optional[GBTensor]:
        """
        Returns an empty copy of the data tensor
        """
        if self._data is None:
            return None
        elif self.is_scalar:
            return Scalar.from_type(self.type_.as_pygrb)
        elif self.is_vector:
            return Vector.sparse(self.type_.as_pygrb, size=self.data.size)
        elif self.is_matrix:
            return Matrix.sparse(self.type_.as_pygrb, nrows=self.data.nrows, ncols=self.data.ncols)
        else:
            raise NotImplementedError("Prototype is only implemented for scalars, vectors, and matrices")

    @property
    def is_scalar(self) -> bool:
        """
        Returns true if the column has zero dimensions, otherwise returns false
        """
        return isinstance(self._data, Scalar)

    @property
    def is_vector(self) -> bool:
        """
        Returns true if the column has one dimension, otherwise returns false
        """
        return isinstance(self._data, Vector)

    @property
    def is_matrix(self) -> bool:
        """
        Returns true if the column has two dimensions, otherwise returns false
        """
        return isinstance(self._data, Matrix)

    @property
    def is_boolean(self) -> bool:
        """
        Returns true if the column has a boolean type, otherwise returns false
        """
        return isinstance(self._type, Types.Boolean)

    @property
    def is_sideloaded(self) -> bool:
        """
        Returns true if the column has sideloaded data
        """
        return self._type.sideload

    @property
    def hidden(self) -> bool:
        """
        Returns true if the column's hidden flag is set, otherwise returns false
        """
        return self._hidden

    @property
    def type_(self) -> Types.Type:
        """
        Returns the column's type
        """
        return self._type

    @property
    def gbtype(self) -> GBType:
        """
        Returns the column's PyGraphBLAS type
        """
        return self._type.as_pygrb

    @property
    def nptype(self) -> np.dtype:
        """
        Returns the column's NumPy type
        """
        return self._type.as_numpy

    @property
    def ndim(self) -> Optional[int]:
        "Returns the number of primary keys in the table"
        if self._data is not None:
          return ndim(self._data)
        else:
          return None

    @property
    def schema(self) -> Dict[Any, Any]:
        return {
            "hidden": self.hidden,
            "type": self.type_.name,
            "empty": self._data is None
        }

    def save_hdf5(self, col_group, pk_cols, verbose=0) -> None:
        """
        Saves column data into h5py gruop
        """
        N = self.ndim
        if N is not None:
            pks = list(tensor_to_numpy(self._data))
            values = pks.pop()

            if self.is_sideloaded:
                if verbose > 0: print("Decoding sideloaded data")
                raw_values = self._refcounts.decode_many(values)
                if verbose > 0: print("Serializing sideloaded data")
                serialized_values = self.type_.serialize(raw_values)
                if verbose > 0: print("Creating array")
                values = np.array(serialized_values, dtype=self.type_.as_numpy, copy=False)

            if verbose > 0: print("Creating hdf5 dataset for value")
            col_group.create_dataset("value", data=values, dtype=self.type_.as_h5py)
            if verbose > 0: print("Creating hdf5 group for primary keys")
            pk_group = col_group.create_group("primary_keys")
            num_digits = len(str(N))
            for it_pk, pk in enumerate(pks):
                if verbose > 0: print("Creating hdf5 dataset for pk", it_pk)
                pk_group.create_dataset(
                  str(it_pk).zfill(num_digits),
                  data = np.array(pk),
                  dtype = pk_cols[it_pk].nptype
                )

    @classmethod
    def load_hdf5(cls, col_group, col_name, shape, **col_info) -> Column:
        """
        Loads column data from an h5py gruop
        """

        col_type = Types.from_string(col_info['type'])

        if not col_info['empty']:
            pk_arrays = list(sorted(col_group['primary_keys'].items(), key=lambda x: x[0]))
            N = len(pk_arrays)
            assert N == len(shape)

            data = []
            for it_pk, (id_pk, pk_arr) in enumerate(pk_arrays):
                data.append(pk_arr)

            values = col_group['value'][:]
            if col_type.sideload:
                refcounts = PyHashOCNT(max(values.size,1)*2)

                values = col_type.deserialize(values)
                values = refcounts.insert_many_numpy(values)
            else:
                refcounts = None
            data.append(values)

            tensor = initialize_tensor(col_type.as_pygrb, shape, data)
        else:
            tensor = None
            refcounts = None

        return Column(col_name, tensor, type_=col_type, hidden=col_info['hidden'], refcounts=refcounts)

    def _insert(self, pk: Optional[Tuple[int]], value: Any):
        if self._data is None:
            raise ValueError("Cannot directly insert primary keys")

        if pk is not None and len(pk) == 0:
            pk = None

        if self.is_scalar:
            assert pk is None
        elif self.is_vector:
            assert len(pk) == 1
        elif self.is_matrix:
            assert len(pk) == 2
        else:
            raise NotImplementedError("Prototype is only implemented for scalars, vectors, and matrices")

        with contextlib.ExitStack() as ocnt_stack:
          if self.is_sideloaded:
              ocnt_stack.enter_context(self._refcounts.checkpoint())
              try:
                current_value_ptr = self._data[pk]
              except KeyError:
                current_value_ptr = 0
              if current_value_ptr != 0:
                  current_value = self._refcounts.decode(current_value_ptr)
                  self._refcounts.remove(current_value)
              value, count = self._refcounts.insert(value)

          if self.is_scalar:
              self._data[None] = value
          elif self.is_vector:
              self._data[pk[0]] = value
          elif self.is_matrix:
              self._data[pk[0], pk[1]] = value

    def _insert_bulk(self, pks: List[np.ndarray], value: np.ndarray) -> None:
        assert isinstance(pks, list)
        assert all(pk.ndim == 1 for pk in pks)
        assert value.ndim == 1
        assert all(isinstance(pk, np.ndarray) for pk in pks)
        assert isinstance(value, np.ndarray)
        assert value.dtype == self.type_.as_numpy
        assert all(pk.size == value.size for pk in pks)

        if self.is_scalar:
            assert len(pks) == 0
        elif self.is_vector:
            assert len(pks) == 1
        elif self.is_matrix:
            assert len(pks) == 2
        else:
            raise NotImplementedError("Prototype is only implemented for scalars, vectors, and matrices")

        with contextlib.ExitStack() as ocnt_stack:
            if self.is_sideloaded:
                ocnt_stack.enter_context(self._refcounts.checkpoint())
                value = self._refcounts.insert_many_numpy(value)
            addtensor = initialize_tensor(self.gbtype, grb_shape(self._data), pks + [value])
            result = self._data.eadd(addtensor, add_op = self.gbtype.FIRST)
            if addtensor.nvals + self._data.nvals != result.nvals:
                raise ValueError("Primary key uniqueness constraint violated")
            else:
                self._data = result







            




















