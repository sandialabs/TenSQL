from __future__ import annotations

import collections
import json
import warnings
from typing import Union, Optional, Sequence, Tuple, Any, Dict

from pygraphblas import Matrix, Vector, Scalar
import pygraphblas.types
import pygraphblas.types
import h5py
import numpy as np

from .Column import Column
from .PrimaryKey import PrimaryKey
from . import util, Types
from .util import GBType, GBTensor, scalar_add, build_tensor, initialize_tensor, grb_shape, ndim, tensor_to_numpy, fill_tensor_from_numpy

class Table:
    """
    Represents a colleciton of tensors, each represented by a column
    """

    def __init__(
        self,
        shape: Sequence[int],
        columns: Sequence[Column],
        stencil_data: GBTensor = None,
    ) -> None:
        assert isinstance(shape, (list, tuple))
        assert len(shape) <= 2
        assert len(shape) <= len(columns)
        assert all(isinstance(d, int) for d in shape)

        assert isinstance(columns, (list, tuple))
        assert all(isinstance(col, Column) for col in columns)

        self._columns = collections.OrderedDict()
        self.shape = tuple(shape)
        # first_column = None

        first_column: Union[None, Column] = None

        for it_column, column in enumerate(columns):
            if it_column >= len(self.shape):
                first_column = column

            if column.name in self._columns:
                raise ValueError(f"Duplicate column name {column.name!r}")

            self._columns[column.name] = column

        # Create the stencil column representing the indices
        if stencil_data is None:
            stencil_data = self._build_tensor(pygraphblas.types.BOOL, self.shape)
            if first_column is not None:
                if not first_column.is_scalar:
                    #TODO: make this the union of all columns, instead of using the first one
                    mask_data = first_column.data.pattern()
                    stencil_data.assign_scalar(True, mask=mask_data)
                elif first_column.data.nvals > 0:
                    stencil_data[None] = True
        else:
            if isinstance(stencil_data, Scalar):
                new_stencil_data = Scalar.from_type(pygraphblas.types.BOOL)
                if stencil_data.nvals > 0 and stencil_data[None]:
                    new_stencil_data[None] = True
                stencil_data = new_stencil_data
            else:
                stencil_data = stencil_data.nonzero()

        name = "_stencil_"
        self._columns[name] = Column(
            name, stencil_data, hidden=True, type_ = Types.Boolean()
        )

        self.mask(stencil_data, inplace=True)

    def __getitem__(self, column_name: str) -> Optional[Column]:
      return self._columns.get(column_name)

    def copy_schema(self):
        new_cols = []
        for name, col in self.columns:
            if col.data is None:
                data = None
            else:
                data = self._build_tensor(col.gbtype, self.shape)
            new_cols.append(Column(name, data, hidden=col.hidden, type_=col.type_))
        return Table(self.shape, new_cols)

    @classmethod
    def from_definition(
        cls,
        column_types: Dict[str, Types.Type],
        primary_keys: Union[Sequence[PrimaryKey], Sequence[str]],
        shape: Optional[Sequence[int, ...]] = None
    ) -> Table:
        assert isinstance(column_types, dict)
        assert isinstance(primary_keys, (tuple, list))

        assert len(primary_keys) <= 2

        columns = []
        default_shape = []
        column_types = column_types.copy()

        for pk_or_name in primary_keys:
            if isinstance(pk_or_name, str):
                coltype = column_types.pop(pk_or_name)
                pk: PrimaryKey = PrimaryKey(pk_or_name, coltype)
            else:
                pk = pk_or_name

            default_shape.append(pk.max_idx)
            columns.append(Column(pk.column_name, None, type_=pk.type_))

        if shape is None:
            shape = default_shape

        shape = tuple(shape)

        for name, type_ in column_types.items():
            gbtype = type_.as_pygrb
            data = cls._build_tensor(gbtype, shape)
            columns.append(Column(name, data, type_=type_))

        return Table(shape, columns)

    @classmethod
    def from_tensor(
        cls, tensor: GBTensor, shape: Optional[Sequence[int]] = None, *, copy=False
    ) -> Table:
        # tensor_shape = util.shape(tensor)
        # if shape is not None:
        #   shape = tuple(shape)
        #   assert len(shape) == tensor_shape
        #   for new_dim, tensor_dim in

        if copy:
            tensor = tensor.dup()

        value_type = Types.from_pygrb(tensor.type)
        assert value_type is not None
        value_gbtype = value_type.as_pygrb

        if isinstance(tensor, Scalar):
            ret = Scalar.from_type(value_gbtype)
            if tensor.nvals > 0:
                ret[None] = tensor[None]

            columns = [Column("value", ret, type_=value_type)]
        elif isinstance(tensor, Vector):
            ret = tensor.apply(value_gbtype.IDENTITY)

            columns = [
                Column("idx", None, type_=Types.BigInt()),
                Column("value", ret, type_=value_type),
            ]
        elif isinstance(tensor, Matrix):
            ret = tensor.cast(value_gbtype)

            columns = [
                Column("ridx", None, type_=Types.BigInt()),
                Column("cidx", None, type_=Types.BigInt()),
                Column("value", ret, type_=value_type),
            ]
        else:
            raise NotImplementedError(
                "Prototype only supports scalars, vectors, and matrices"
            )

        return Table(util.shape(ret), columns)

    @property
    def schema(self):
        return {
            "nnz": self.count(),
            "shape": self.shape,
            "columns": [
                [c_name, c.schema]
                for c_name, c in self.columns
            ],
        }

    @property
    def ndim(self) -> int:
        "Returns the number of primary keys in the table"
        return len(self.shape)

    @property
    def is_scalar(self) -> bool:
        "Returns true iff the table has 0 primary keys"
        return len(self.shape) == 0

    @property
    def is_vector(self) -> bool:
        "Returns true iff the table has 1 primary keys"
        return len(self.shape) == 1

    @property
    def is_matrix(self) -> bool:
        "Returns true iff the table has 2 primary keys"
        return len(self.shape) == 2

    def is_primary_key(self, check_col: Column):
        "Checks whether a given column is a primary key column"
        for it_col, (col_name, col) in enumerate(self.columns):
            if it_col >= self.ndim:
                break
            if col is check_col:
                return True
        return False

    @property
    def primary_key_names(self) -> Tuple[str, ...]:
        "Returns the names of the table's primary keys"
        return tuple(col_name for it_col, (col_name, col) in enumerate(self.columns) if it_col < self.ndim)

    @property
    def stencil(self) -> GBTensor:
        "Returns the sparsity pattern of the table defining the rows"
        return self._columns["_stencil_"].data

    def get_column(self, name: str) -> Optional[Column]:
        "Returns a specific column associated with the given name (or None)"
        return self._columns.get(name)

    @property
    def columns(self) -> Sequence[Tuple[str, Column]]:
        "Generates a sequence of (column_name, column) pairs"
        for name, col in self._columns.items():
            if not col.hidden:
                yield name, col

    @classmethod
    def _build_tensor(cls, gbtype: pygraphblas.types.Type, shape: Sequence[int]) -> GBTensor:
        return build_tensor(gbtype, shape)

    @classmethod
    def _initialize_tensor(
        cls, gbtype: pygraphblas.types.Type, shape: Sequence[int], lists: Sequence[Sequence[Any]]
    ) -> GBTensor:
        return initialize_tensor(gbtype, shape, lists)

    def count(self) -> int:
        return self.stencil.nvals

    def add(self, other: Table, *, inplace=False, op: Any = None) -> Table:
        assert self.shape == other.shape
        assert len(self._columns) == len(other._columns)

        new_cols = []
        for self_col, other_col in zip(self._columns.values(), other._columns.values()):
            assert self_col.name == other_col.name
            assert (self_col.data is None) == (other_col.data is None)

            # if self_col.name == "_stencil_":
            #  continue

            if self_col.data is not None:
                if self.is_scalar:
                    data = scalar_add(self_col.data, other_col.data, op=op, inplace=inplace)
                elif not inplace:
                    data = self_col.data.eadd(other_col.data, add_op=op)
                elif inplace:
                        data = self_col.data.eadd(
                            other_col.data, add_op=op, out=self_col.data
                        )
            else:
                data = None

            if not inplace:
                new_cols.append(
                    Column(
                        self_col.name,
                        data,
                        type_=self_col.type_,
                        hidden=self_col.hidden,
                    )
                )

        if not inplace:
            return Table(self.shape, new_cols)
        else:
            return self

    def mask(self, mask: GBTensor, *, inplace=False):
        new_cols = []

        if not self.is_scalar:
            mask = self.stencil.emult(mask).nonzero()
        else:
            if self.stencil.nvals == 0:
                mask.clear()

        for it, (cname, col) in enumerate(self.columns):
            if col.data is not None:
                if not inplace:
                    data = self._build_tensor(col.type_, self.shape)
                else:
                    data = col.data

                if self.is_scalar:
                    if mask.nvals > 0 and col.data.nvals > 0:
                        data[0] = col.data[0]
                    else:
                        data.clear()
                else:
                    data.assign(col.data, mask=mask)
            else:
                data = None
            new_cols.append(Column(cname, data, type_=col.type_))

        if not inplace:
            return Table(self.shape, new_cols, stencil_data=mask)
        else:
            name = "_stencil_"
            self._columns[name] = Column(
                name, mask, hidden=True, type_=Types.Boolean()
            )
            return self

    def insert(self, **kwargs: Dict[str, Any]) -> None:
        assert set(kwargs.keys()) <= set(
            cname for cname, col in self._columns.items() if not col.hidden
        )
        pk = []
        for it, (cname, col) in enumerate(self.columns):
            if it >= self.ndim:
                break
            pk.append(kwargs[cname])
        
        if self.is_scalar:
            self.stencil[None] = True
            for cname, col in list(self.columns)[self.ndim :]:
                if cname in kwargs:
                    col._insert(None, kwargs[cname])
            return
        elif self.is_vector:
            # assign_scalar only works with slices on vectors
            pk = tuple(slice(idx, idx) for idx in pk)
        elif self.is_matrix:
            pk = tuple(pk)
        self.stencil.assign_scalar(True, *pk)
        for cname, col in list(self.columns)[self.ndim :]:
            if cname in kwargs:
                col._insert(pk, kwargs[cname])
        return

    def insert_from_numpy(self, data: np.ndarray) -> None:
        nptype = data.dtype
        colnames = set(c_name for c_name, column in self.columns)
        assert isinstance(nptype, np.dtype)
        assert all(nptype_name in colnames for nptype_name in nptype.names)
        assert all(nptype.fields[c_name][0] == column.nptype for c_name, column in self.columns)

        #TODO: rollback all changes to all columns and the stencil on failure
        pks = [data[c_name] for itdim, (c_name, column) in enumerate(self.columns) if itdim < self.ndim]
        stencil_data = pks + [[True] * data.size]
        add_stencil = self._initialize_tensor(self.stencil.type, grb_shape(self.stencil), stencil_data)
        self.stencil.eadd(add_stencil, out=self.stencil)

        for itdim, (c_name, column) in enumerate(self.columns):
            if itdim < self.ndim:
                pass
            else:
                column._insert_bulk(pks, data[c_name])


    def __len__(self) -> int:
        return self.stencil.nvals

    def to_records(self) -> np.ndarray:
        names, columns = zip(*self.columns)
        nvals, ndim = self.stencil.nvals, self.ndim

        #ret_dtype = [(name, column.type_.as_numpy) for name, column in self.columns]
        ret_dtype = [(name, object) for name, column in self.columns]
        ret = np.empty(shape=(nvals,), dtype=ret_dtype)

        stencil = self.stencil
        indices = tensor_to_numpy(stencil)[:-1]
        offsets = np.arange(stencil.nvals, dtype=np.int64)

        all_offsets_tensor = build_tensor(pygraphblas.types.INT64, self.shape)
        fill_tensor_from_numpy(all_offsets_tensor, *indices, offsets)

        if ndim == 1:
            ret[names[0]] = indices[0]
        elif ndim == 2:
            ret[names[0]], ret[names[1]] = indices[0], indices[1]

        for it_column, column in enumerate(columns):
            tensor = column.data
            if it_column < self.ndim:
                continue
            
            if ndim == 1:
                #TODO Cast mask to BOOL
                offset_tensor = all_offsets_tensor.extract(tensor)
            elif ndim == 2:
                bool_tensor = Matrix.sparse(typ=pygraphblas.types.BOOL, nrows=tensor.nrows, ncols=tensor.ncols)
                tensor.apply(tensor.type.IDENTITY, out=bool_tensor)
                offset_tensor = all_offsets_tensor.extract_matrix(mask=bool_tensor)

            offset_indices = list(tensor_to_numpy(offset_tensor))
            offset_values = offset_indices.pop()

            tensor_indices = list(tensor_to_numpy(tensor))
            tensor_values = tensor_indices.pop()

            #print(f"{offset_tensor.nvals=}, {tensor.nvals=}, {all_offsets_tensor.nvals=}, {self.stencil.nvals=}")

            #for off_idx, ten_idx in zip(offset_indices, tensor_indices):
            #    np.testing.assert_equal(off_idx, ten_idx)

            if column.is_sideloaded:
                tensor_values = column._refcounts.decode_many(tensor_values)

            ret[names[it_column]][offset_values] = tensor_values

        return ret


    def __iter__(self):
        if self.is_scalar:
            if self.stencil.nvals > 0:
                row = {}
                for it_col, (name, c) in enumerate(self.columns):
                    if c.data.nvals == 0:
                        value = None
                    elif c.is_sideloaded:
                        value = c._refcounts.decode(c.data[None])
                    else:
                        value = c.data[None]
                    row[name] = value
                yield row
        elif self.is_vector or self.is_matrix:
            ret = collections.OrderedDict()
            pk_names = [name for name, c in list(self.columns)[: self.ndim]]

            all_null = {name: None for it_col, (name, c) in enumerate(self.columns) if it_col >= self.ndim}

            stencil_lists = self.stencil.to_lists()
            for row in zip(*stencil_lists[:-1]):
                new_row = dict(zip(pk_names, row))
                new_row.update(all_null)
                ret[tuple(row)] = new_row

            for it_col, (name, c) in enumerate(self.columns):
                if it_col >= self.ndim:
                    col_lists = c.data.to_lists()
                    for col_row in zip(*col_lists):
                        pks, val = col_row[:-1], col_row[-1]
                        row = ret.get(pks)
                        if row is None:
                            pass
                        elif c.is_sideloaded:
                            row[name] = c._refcounts.decode(val)
                        else:
                            row[name] = val
            
            yield from ret.values()
        else:
            raise NotImplementedError(
                "Prototype only supports scalars, vectors, and matrices"
            )

    def save_hdf5(self, table_group, verbose=0) -> None:
        table_group.attrs.create(
            "table", json.dumps(self.schema), shape=tuple(), dtype=h5py.string_dtype()
        )

        if self.ndim == 0:
            idx_list = []
        else:
            idx_list = self.stencil.to_lists()[: self.ndim]

        data_group = table_group.create_group("data")
        stencil_group = table_group.create_group("stencil")
        pk_cols = []

        for it, (col_name, col) in enumerate(self.columns):

            if it < self.ndim:
                if verbose >= 1:
                    print("Saving", "primary key", col_name)
                values = np.array(idx_list[it], dtype=col.nptype)
                stencil_group.create_dataset(col_name, data=values)
                pk_cols.append(col)
            else:
                if verbose >= 1:
                    print("Saving", "column", col_name, "with", col.data.nvals, "nnz")
                col_group = data_group.create_group(col_name)
                col.save_hdf5(col_group, pk_cols, verbose=verbose-1)

    @classmethod
    def load_hdf5(cls, table_group) -> Table:
        metadata = json.loads(table_group.attrs["table"])
        shape = metadata["shape"]
        nnz = metadata['nnz']
        ndim = len(shape)
        columns = []

        idx_list = []
        data_group = table_group["data"]
        stencil_group = table_group["stencil"]
        for it, (col_name, col_info) in enumerate(metadata["columns"]):
            if it < ndim:
                column = Column.load_hdf5(None, col_name, shape, **col_info)
                idx_list.append(stencil_group[col_name])
            else:
                column = Column.load_hdf5(data_group[col_name], col_name, shape, **col_info)

            columns.append(column)

        if nnz > 0:
            stencil_data = cls._initialize_tensor(
                pygraphblas.types.BOOL, shape, idx_list + [[True] * nnz]
            )
        else:
            stencil_data = cls._build_tensor(pygraphblas.types.BOOL, shape)

        return Table(shape, columns, stencil_data)
    
    def __del__(self) -> None:
        pass
#        for it_col, (name, c) in enumerate(self.columns):
#            if it_col >= self.ndim:
#                if c.type_.sideload:
#                    col_lists = c.data.to_lists()
#                    for val in col_lists[-1]:
#                        c.type_.sideload_hash.remove_by_id(val)
