from __future__ import annotations
import collections
import itertools

from typing import Optional, Sequence, Callable, Tuple, Any, Dict, FrozenSet

import pygraphblas.types
from pygraphblas.types import BOOL as GBBool
from pygraphblas import Matrix, Vector, Scalar

from .. import Types
from ..PrimaryKey import PrimaryKey
from ..Column import Column
from ..Table import Table
from ..util import SequentialIdGenerator, EncoderDecoder
from .. import LAIR

Promotion = Callable[[LAIR.Node], LAIR.Node]
QualifiedColumnName = Tuple[str, str] # Qualified Name: (TableName, ColumnName)
JoinMatches = FrozenSet[QualifiedColumnName]
AxesSpecifier = Tuple[int, ...]

def PromoteStencil(
    pattern: Tuple[AxesSpecifier, AxesSpecifier, AxesSpecifier],
    A: LAIR.Node, 
    B: LAIR.Node, 
    kind: str
):
    NDIM = tuple()
    if kind == 'inner':
        if pattern == (NDIM, NDIM, NDIM):
            oStencil = LAIR.ElementwiseMultiply(A, B)
        elif pattern[0] == NDIM:
            if x.nvals > 0 and y.nvals > 0:
                oStencil = x.apply_second(grb_op, y)
            elif isinstance(x, Vector):
                oStencil = Vector.sparse(x.type, x.size)
            elif isinstance(x, Matrix):
                oStencil = Matrix.sparse(x.type, x.nrows, x.ncols)

class JoinPromotion:
    def __init__(self, idx_in: AxesSpecifier, idx_out: AxesSpecifier, stencil: LAIR.Node):
        self.idx_in = idx_in
        self.idx_out = idx_out
        self.stencil = stencil

    def __call__(self, node: LAIR.node) -> LAIR.Node:
        return LAIR.InnerBroadcastMask(
            node, self.idx_in,
            self.stencil, self.idx_out,
            self.idx_out
        )

class TableBuilder:
    def __init__(
        self,
        shape: Sequence[int],
        stencil_builder: LAIR.Node,
        column_builders: Dict[str, Optional[LAIR.Node]],
    ):
        self._shape = tuple(shape)
        self._stencil_builder = stencil_builder
        self._column_builders = column_builders

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def stencil(self) -> LAIR.Node:
        return self._stencil_builder

    def get_column(self, name):
        #print(tuple(self._column_builders.keys()))
        return self._column_builders[name]

    def mask(self, mask: LAIR.Node) -> TableBuilder:
        out_cols = collections.OrderedDict()
        for cb_name, cb in self._column_builders.items():
            if cb is not None:
                out_cols[cb_name] = LAIR.Mask(cb, mask)
            else:
                out_cols[cb_name] = None
        return TableBuilder(self._shape, LAIR.Mask(self._stencil_builder, mask), out_cols)


class JoinerResult:
    def __init__(
        self,
        tables: Dict[str, TableBuilder],
        common_stencil: LAIR.Node,
        common_shape: Sequence[int],
        common_keys: Sequence[JoinMatches],
    ):
        self._tables = tables
        self._stencil = common_stencil
        self._shape = tuple(common_shape)
        self._keys = tuple(common_keys)

    @property
    def tables(self) -> Dict[str, TableBuilder]:
        return self._tables

    @property
    def stencil(self) -> LAIR.Node:
        return self._stencil

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def primary_keys(self) -> Tuple[JoinMatches, ...]:
        return self._keys

    def mask(self, mask: LAIR.Node) -> JoinerResult:
        new_tables = {k: v.mask(mask) for k, v in self._tables.items()}
        new_stencil = LAIR.Mask(self._stencil, mask)
        return JoinerResult(new_tables, new_stencil, self._shape, self._keys)


class Joiner:
    INDEX_TYPE = Types.BigInt()

    def __init__(
        self,
        out_shape: Sequence[int],
        stencil: LAIR.Node,
        promote_left: Promotion,
        promote_right: Promotion,
        join_keys: Sequence[JoinMatches],
    ):
        join_keys = tuple(join_keys)
        out_shape = tuple(out_shape)
        assert len(join_keys) == len(out_shape)

        self._join_keys = join_keys
        self._out_shape = out_shape
        self._stencil = stencil
        self._promote_left = promote_left
        self._promote_right = promote_right

    @property
    def ndim(self) -> int:
        return len(self._out_shape)

    def promoteTable(self, alias: str, table: TableBuilder, *, is_left: bool) -> TableBuilder:
        if is_left:
            promote = self._promote_left
        else:
            promote = self._promote_right

        table_pk_qnames = []
        for itdim, (col_name, _) in enumerate(table._column_builders.items()):
            if itdim < table.ndim:
                table_pk_qnames.append((alias, col_name))

        out_cols = collections.OrderedDict()
        for itdim, dim_keys in enumerate(self._join_keys):
            match_cols = [qname for qname in table_pk_qnames if qname in dim_keys]
            if len(match_cols) == 0:
                col_name = f"__bc_pk_{itdim}__"
            elif len(match_cols) == 1:
                table_alias, col_name = match_cols[0]
            else:
                raise ValueError(
                    "Found multiple names for output primary key (join by tensor diagonal)"
                )
            if len(match_cols) == 0:
              pk = PrimaryKey(col_name, self.INDEX_TYPE)
              out_cols[col_name] = LAIR.PrimaryKey(self._stencil, itdim, pk.gbtype, name=col_name)

        for it, (col_name, in_col) in enumerate(table._column_builders.items()):
            #if it >= table.ndim:
            out_cols[col_name] = promote(in_col)

        return TableBuilder(self._out_shape, self._stencil, out_cols)

    def joinAdditionalTable(
        self, left: JoinerResult, right_alias: str, right: TableBuilder
    ) -> JoinerResult:
        tables = {}
        for table_alias, table in left.tables.items():
            assert table_alias not in tables
            tables[table_alias] = self.promoteTable(table_alias, table, is_left=True)

        assert right_alias not in tables
        tables[right_alias] = self.promoteTable(right_alias, right, is_left=False)

        return JoinerResult(tables, self._stencil, self._out_shape, self._join_keys)

    @classmethod
    def from_axes(
        cls,
        A: LAIR.Node,
        aCols: Sequence[QualifiedColumnName], #Qualified names of a's columns
        aShape: Sequence[int],
        B: LAIR.Node,
        bCols: Sequence[QualifiedColumnName], #Qualified names of b's columns
        bShape: Sequence[int],
        join_keys: Sequence[JoinMatches],
        kind: str = "inner",
    ) -> Joiner:
        aCols, aShape = tuple(aCols), tuple(aShape)
        bCols, bShape = tuple(bCols), tuple(bShape)

        #print(f"{aCols=}, {aShape=}, {bCols=}, {bShape=}, {join_keys=}")

        assert kind in {"inner", "outer", "left", "right"}
        if kind != "inner":
            raise NotImplementedError()  # Left, Right, and Outer broadcasts

        if len(aCols) < len(bCols):
            aCols, aShape, A, bCols, bShape, B = bCols, bShape, B, aCols, aShape, A
            reversed = True
            if kind == "left":
                kind = "right"
            elif kind == "right":
                kind = "left"
        else:
            reversed = False

        join_keys = tuple(join_keys)

        # Maps qualified names of relevant primary keys to integer ids
        all_idx = EncoderDecoder(SequentialIdGenerator())
        oIndices = []
        oCols = []
        for join_matches in join_keys:
            match_id: Optional[int] = None
            for qname in join_matches:
                if match_id is None:
                    match_id = all_idx[qname]
                    oCols.append(qname)
                else:
                    all_idx[qname] = match_id
            assert match_id is not None
            oIndices.append(match_id)

        aIndices = tuple(all_idx[col] for col in aCols)
        bIndices = tuple(all_idx[col] for col in bCols)
        oIndices = tuple(oIndices)

        # Does not support reductions or broadcasting to new axes
        assert set(oIndices) == set(aIndices) | set(bIndices)

        # Build the result shape
        oShape = [None for oidx in oIndices]
        for name, idx, dim in itertools.chain(zip(aCols, aIndices, aShape), zip(bCols, bIndices, bShape)):
            if idx >= len(oShape):
                continue
            elif oShape[idx] is None:
                oShape[idx] = dim
            elif dim != oShape[idx]:
                oname = oCols[idx]
                raise ValueError(
                    "Shape operand dimension {name} disagrees with expected dimension for output primary key {oname}"
                )
        #print(f"{oShape=}")
        assert None not in oShape

        if kind == "inner":
#            oStencil: LAIR.Node = LAIR.ElementwiseMultiply(
#                'LAND',
#                LAIR.Promote(A, aIndices, oIndices),
#                LAIR.Promote(B, bIndices, oIndices),
#            )
            oStencil: LAIR.Node = LAIR.InnerBroadcast("LAND", A, aIndices, B, bIndices, oIndices)
#        elif kind == "outer":
#            oStencil: LAIR.Node = LAIR.BroadcastOuter(A, aIndices, B, bIndices, oIndices)
        promote_left: JoinPromotion = JoinPromotion(aIndices, oIndices, oStencil)
        promote_right: JoinPromotion = JoinPromotion(bIndices, oIndices, oStencil)

        if reversed:
            promote_left, promote_right = promote_right, promote_left

        return Joiner(oShape, oStencil, promote_left, promote_right, join_keys)
