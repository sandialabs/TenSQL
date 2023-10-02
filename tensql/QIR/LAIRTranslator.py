from __future__ import annotations

import collections
import contextlib
from typing import Any, List, Dict, Tuple, Sequence, FrozenSet, Optional
import itertools
import json
import sys

from .Table import QirTable
from .QirColumn import QirColumn

from .. import QIR, LAIR, Table, Database
from ..Column import Column
from ..util import SequentialIdGenerator, EncoderDecoder, grb_shape, tensor_to_numpy
from . import Joiner
from .Joiner import JoinerResult, JoinMatches
from .Visitor import Visitor
from ..LAIR.Optimizer import optimize
from .. import Types
from .._PyHashOCNT import PyHashOCNT

import pygraphblas
import pygraphblas.types
from pygraphblas import Scalar, Vector, Matrix

def _get_reduce_pattern(
    input_matches: Tuple[JoinMatches, ...], output_keys: Tuple[JoinMatches, ...]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    NODIM = tuple()
    if len(input_matches) == 0:
        pattern = (NODIM, NODIM)
    elif len(input_matches) == 1:
        if len(output_keys) == 0:
            pattern = ((0,), NODIM)
        else:
            pattern = ((0,), (0,))
    elif len(input_matches) == 2:
        if len(output_keys) == 0:
            pattern = ((0, 1), NODIM)
        elif len(output_keys) == 2:
            pattern = ((0, 1), (0, 1))
        else:
            if any(qname in input_matches[0] for qname in output_keys[0]):
                pattern = ((0, 1), (0,))
            elif any(qname in input_matches[1] for qname in output_keys[0]):
                pattern = ((0, 1), (1,))
            else:
                assert False
    else:
        all_idx = {}
        for it_idx, idx_matches in enumerate(input_matches):
            for it, qname in enumerate(idx_matches):
                all_idx[qname] = it_idx

        ret = []
        for it_idx, idx_matches in enumerate(output_keys):
            matches = []
            match_ids = set()
            for it, qname in enumerate(idx_matches):
                match_id = all_idx.get(qname)
                if match_id is not None:
                    matches.append(qname)
                    match_ids.add(match_id)
                    break
            if len(match_ids) == 0:
                raise ValueError(
                    f"All output keys for dimension {it_idx} missing from input keys: {idx_matches}"
                )
            elif len(match_ids) > 1:
                raise ValueError(
                    f"Output keys for dimension {it_idx} matched input keys in multiple dimensions: {match_ids}"
                )
            else:
                ret.append(list(match_ids)[0])
        pattern = (tuple(range(len(input_matches))), tuple(ret))
    return pattern


class SelectVisitor(Visitor):
    def visit(self, node: QIR.ABCNode) -> LAIR.Node:
        return super().visit(node)

    def __init__(self, join_result: JoinerResult, expected_keys: Sequence[JoinMatches]):
        self._join_result = join_result
        self._expected_keys = tuple(expected_keys)
        self._refcounts = PyHashOCNT(16)

    @property
    def join_map(self) -> Dict[str, Table]:
        return self._join_result.tables

    def generic_visit(self, node: QIR.ABCNode) -> None:
        classes = [cls for cls in type(node).mro() if issubclass(cls, QIR.ABCNode)]

        raise NotImplementedError(
            f"{type(self).__name__} does not implement a visit method for any of {classes}"
        )

    def visit_QirColumn(self, node: QIR.QirColumn) -> LAIR.Node:
        return self.join_map[node.table.alias].get_column(node.raw.name)

    def visit_Constant(self, node: QIR.Constant) -> LAIR.Node:
        type_ = Types.from_native(type(node.value))
        ret = Scalar.from_type(type_.as_pygrb)
        if type_.sideload:
            ret[None], count = self._refcounts.insert(node.value)
        else:
            ret[None] = node.value

        out_idx = tuple(range(self._join_result.ndim))

        return LAIR.InnerBroadcastMask(
            LAIR.Tensor(ret, name=node.description, stencil=None), tuple(),
            self._join_result.stencil, out_idx,
            out_idx
        )
            
#        return LAIR.Tensor(ret, name=node.description, stencil=None)
          

#        stencil = Scalar.from_type(Types.Boolean._as_pygrb)
#        stencil[None] = True
#        stencil = LAIR.Tensor(
#          stencil, name=f"_constant.stencil", stencil=None
#        )
#
#        shape = tuple()
#        columns = collections.OrderedDict()
#        columns["value"] = LAIR.Tensor(
#            ret, name="_constant.value", stencil=stencil
#        )
#
#        table_builder = Joiner.TableBuilder(shape, stencil, columns)
#
#
#        return Joiner.TableBuilder(shape, stencil, columns)


    def visit_ABCUnaryExpression(self, node: QIR.ABCUnaryExpression) -> LAIR.Node:
        x = self.visit(node.operand)
        return LAIR.ElementwiseApply(node.op, x)

    def visit_Cast(self, node: QIR.Cast) -> LAIR.Node:
        x = self.visit(node.operand)
        return LAIR.Cast(x, node.type_.as_pygrb)

    def visit_ArithmeticBinaryExpression(
        self, node: QIR.ArithmeticBinaryExpression
    ) -> LAIR.Node:
        x = self.visit(node.left)
        y = self.visit(node.right)

        # (null + anything) = null, so everything equates to EWiseMult
        return LAIR.ElementwiseMultiply(node.op, x, y)

    def visit_ABCReductionExpression(
        self, node: QIR.ABCReductionExpression
    ) -> LAIR.Node:
        x = self.visit(node.operand)

        pattern = _get_reduce_pattern(
            self._join_result.primary_keys, self._expected_keys
        )
        #print(self._join_result.primary_keys, self._expected_keys, pattern)

        return LAIR.Reduction(node.op, x, set(pattern[0]) - set(pattern[1]))

    def visit_Count(self, node: QIR.Count) -> LAIR.Node:
        x = self.visit(node.operand)
        pattern = _get_reduce_pattern(
            self._join_result.primary_keys, self._expected_keys
        )
        x1 = LAIR.ElementwiseApply("ONE", x)

        return LAIR.Reduction(node.op, x1, set(pattern[0]) - set(pattern[1]))

    def visit_BooleanBinaryExpression(
        self, node: QIR.BooleanBinaryExpression
    ) -> LAIR.Node:
        x = self.visit(node.left)
        y = self.visit(node.right)

        # (null || anything) = null, so everything equates to EWiseMult
        return LAIR.ElementwiseMultiply(node.op, x, y)


from .Visitor import JoinPlan, JoinPlanOnClauseVisitor, JoinPlanVisitor


class FromVisitor(Visitor):
    def __init__(self, db: Database, outkeys: Sequence[QIR.ArithmeticColumn]):
        self.db = db
        self._outkeys = outkeys

    def visit_From(self, node: QIR.From) -> JoinerResult:
        return self.visit(node.table)

    def to_TableBuilder(self, node: QIR.QirTable) -> Joiner.TableBuilder:
        stencil = LAIR.Tensor(
            node.raw.stencil, name=f"{node.alias}.stencil", stencil=None
        )
        shape = node.raw.shape
        columns = collections.OrderedDict()
        for it, col in enumerate(node.columns()):
            if it < len(shape):
                columns[col.name] = LAIR.PrimaryKey(stencil, it, col.raw.type_, name=".".join(col.qualname))
            else:
                columns[col.name] = LAIR.Tensor(
                    col.raw.data, name=".".join(col.qualname), stencil=stencil
                )

        return Joiner.TableBuilder(shape, stencil, columns)

    def visit_QirTable(self, node: QIR.QirTable) -> JoinerResult:
        grb_table = self.to_TableBuilder(node)

        tables = {node.alias: grb_table}

        primary_keys = [{(pk.table.alias, pk.name)} for pk in node.primary_keys]
        #print(f"Building table {node.alias} with primary keys: {primary_keys}")

        return JoinerResult(
            tables=tables,
            common_stencil=grb_table.stencil,
            common_shape=grb_table.shape,
            common_keys=primary_keys,
        )

    def visit_Join(self, node: QIR.Join) -> JoinerResult:
        left_result = self.visit(node.left)
        join_plan = JoinPlanVisitor().visit(node)
        join_keys = join_plan.realize(self._outkeys)

        sA = left_result.stencil
        aidx = [tuple(ax)[0] for ax in left_result.primary_keys]
        ashape = left_result.shape
        B = self.to_TableBuilder(node.right)
        sB = B.stencil
        bidx = [(col.table.alias, col.name) for col in node.right.primary_keys]
        bshape = node.right.raw.shape

        joiner = Joiner.Joiner.from_axes(sA, aidx, ashape, sB, bidx, bshape, join_keys)
        return joiner.joinAdditionalTable(left_result, node.right.alias, B)

    def visit(self, node: QIR.ABCNode) -> JoinerResult:
        return super().visit(node)


class WhereVisitor(SelectVisitor):
    pass


class QueryVisitor(Visitor):
    def __init__(self, db: Database):
        self.db = db

    def visit_SelectQuery(self, node: QIR.SelectQuery) -> Table:
        ################################################
        # Construct the join plan
        ################################################

        QIR.ABCNode.save_graph("QIR-viz.dot", node)

        if node._group_by is not None:
            outkeys = node._group_by.columns
        else:
            outkeys = []
            for clause in node._selects:
                if isinstance(
                    clause, QIR.ArithmeticColumn
                ) and clause.table.raw.is_primary_key(clause.raw):
                    outkeys.append(clause)
                else:
                    break

        #print(f"{outkeys=}")
        join_result = FromVisitor(self.db, outkeys).visit(node._table_seq)

        if node._group_by is not None:
            expected_primary_keys = [{col.qualname} for col in node._group_by.columns]
        else:
            expected_primary_keys = join_result.primary_keys

        #print(f"{expected_primary_keys=}")

        if node._where is not None:
            where = WhereVisitor(join_result, expected_primary_keys).visit(node._where)
            join_result = join_result.mask(LAIR.Pattern(where))

        ############################################
        # Create pseudo-columns for primary keys
        ############################################
        columns = []
        column_names = []
        #print(f"{expected_primary_keys=}")
        #print(f"{outkeys=}")
        for pk_aliases, pk_select in itertools.zip_longest(
            expected_primary_keys, node._selects
        ):
            if pk_aliases is None:
                # Select clauses after primary keys
                if isinstance(
                    pk_select, QIR.ArithmeticColumn
                ) and pk_select.table.raw.is_primary_key(pk_select.raw):
                    select_qname = pk_select.qualname
                    raise ValueError(
                        f"Unexpected additional primary key discovered in select clause: {select_qname}."
                    )
            else:
                if (
                    pk_select is None
                    or not isinstance(pk_select, QIR.ArithmeticColumn)
                    or not pk_select.table.raw.is_primary_key(pk_select.raw)
                ):
                    #print(f"{pk_select is None=},{not isinstance(pk_select, QIR.ArithmeticColumn)=},{not pk_select.table.raw.is_primary_key(pk_select.raw)=},{pk_aliases=},{pk_select=}")
                    expected = set(pk_aliases)
                    if pk_select is None:
                        observed = None
                    else:
                        observed = pk_select.qualname
                    raise ValueError(
                        f"Query select statements must start with the correct primary keys in the correct order.  Expected one of: {expected}, but found {observed}."
                    )

                select_qname = pk_select.qualname
                if select_qname not in pk_aliases:
                    raise ValueError(
                        f"Query select statements must start with the correct primary keys in the correct order.  Found: {select_qname}, but expected one of: {set(pk_aliases)}"
                    )
                columns.append(
                    Column(pk_select.alias, None, type_=pk_select.raw.type_)
                )
            column_names.append(pk_select.name)

        ################################################
        # Generate the IR for the output columns
        ################################################
        if node._group_by is not None:
            # Handle Reductions
            reduce_pattern = _get_reduce_pattern(
                join_result.primary_keys, expected_primary_keys
            )

            axes = [x for x in reduce_pattern[0] if x not in reduce_pattern[1]]
            out_ndim = join_result.ndim - len(axes)

            #print(f"Reduction: {axes=}, {out_ndim=}")
            out_stencil = LAIR.Reduction("LOR", join_result.stencil, axes)
        else:
            out_ndim = join_result.ndim
            out_stencil = join_result.stencil

        column_builders = []
        for col_name, col in zip(column_names[out_ndim:], node._selects[out_ndim:]):
            column_builders.append(
                LAIR.Output(
                    SelectVisitor(join_result, expected_primary_keys).visit(col),
                    name = col.alias
                )
            )

        out_stencil = LAIR.Output(out_stencil, "stencil")

        ################################################
        # Optimize the query
        ################################################

        LAIR.Node.save_graph("IRViz-unoptimized.dot", *column_builders, out_stencil)
        column_builders = [optimize(column) for column in column_builders]
        out_stencil = optimize(out_stencil)
        LAIR.Node.save_graph("IRViz-optimized.dot", *column_builders, out_stencil)

        ################################################
        # Execute the intermediate representation
        ################################################
        from ..LAIR.Visitor import ExpressionVisitor as LAIRVisitor

        column_tensors = list(LAIRVisitor().run(*column_builders, out_stencil))
        out_stencil = column_tensors.pop()
        #print(f"{len(grb_shape(out_stencil))=},{out_ndim=}")
        assert len(grb_shape(out_stencil)) == out_ndim
        #print("Finished executing LAIR") ; sys.stdout.flush()
        #print("column_builder names", [col.name for col in column_builders])
        #print("select names", [qircol.alias for qircol in node._selects[out_ndim:]])
        #print("column tensors", [tensor.type for tensor in column_tensors])
        #print("column tensors from_grb", [Types.from_pygrb(tensor.type) for tensor in column_tensors])


        for qircol, tensor in zip(node._selects[out_ndim:], column_tensors):
            type_ = Types.from_pygrb(tensor.type)
            #print("Building column", qircol.alias, type_) ; sys.stdout.flush()
            col = Column(qircol.alias, tensor, type_=type_)
            if col.data is not None and col.is_sideloaded:
                as_lists = tensor_to_numpy(col.data)
                col._refcounts.insert_many_encoded_numpy(as_lists[-1])
            columns.append(col)

        return Table(grb_shape(out_stencil), columns, stencil_data=out_stencil)

    def visit_InsertQuery(self, node: QIR.InsertQuery) -> Table:
        ndim = node.table.raw.ndim
        from_table = node._from_table
        from_records = node._from_records

        if from_records is not None:
            table = node.table.raw.copy_schema()
            table.insert_from_numpy(from_records)
            from_table = QIR.QirTable(node.db, "__FROM_RECORDS__", table)

        outdata = {}
        required_keys = set()
        available_keys = set()
        for it, (cname, col) in enumerate(node.table.raw.columns):
            available_keys.add(cname)
            if it >= ndim:
              outdata[cname] = tuple([[] for _ in range(ndim)] + [[]])
            else:
              required_keys.add(cname)

        #print("Building tensors")  ; sys.stdout.flush()
        stencildata = [[] for _ in range(ndim)]
        nvals = 0
        for it_row, row in enumerate(node._values):
            assert len(row) >= len(required_keys) and all(k in row for k in required_keys)
            assert not any(k not in available_keys for k in row)

            nvals += 1

            for it, (cname, col) in enumerate(node.table.raw.columns):
                # Get the value or its default
                default = None
                value = row.get(cname, default)

                if it < ndim:
                    # Add the primary key value to the stencil data
                    assert isinstance(value, int) or isinstance(value, np.integer)
                    stencildata[it].append(value)
                elif value is not None:
                    # Add the value to the output lists for the column
                    for it in range(ndim): # Append primary keys
                        outdata[cname][it].append(stencildata[it][-1])
                    if col.is_sideloaded:
                        value = id(value)
                    outdata[cname][ndim].append(value) # Append the value
                else:
                    pass
        #print("Done building tensors") ; sys.stdout.flush()

        
        column_builders = []

        ################################################
        # Execute the intermediate representation
        ################################################
        from ..LAIR.Visitor import ExpressionVisitor as LAIRVisitor
        from ..LAIR.ElementwiseAdd import ElementwiseAdd
        from ..LAIR.Tensor import Tensor

        if from_table is None:
            stencildata.append([True] * nvals)
            addstencil = Table._initialize_tensor(
                Types.Boolean.as_pygrb,
                node.table.raw.shape,
                stencildata
            )
            # Ensure the primary key constraints aren't violated by the incoming data
            if (not node._if_not_exists) and (addstencil.nvals != nvals):
                raise ValueError("Primary key uniqueness constraint violated")
        else:
            addstencil = from_table.raw.stencil
        addstencil = Tensor(addstencil, name='INSERT_VALUES.stencil')

        table_stencil = Tensor(node.table.raw.stencil, name=f"{node.table.name}.stencil")
        if node._if_not_exists:
            # Don't upload stuff that is already in the database
            addstencil = LAIR.AntiMask(addstencil, table_stencil)

        refcount_sources = {}

        #print("Building LAIR")  ; sys.stdout.flush()
        for it, (cname, col) in enumerate(node.table.raw.columns):
            if it >= ndim:
                if from_table is None:
                    adddata = Table._initialize_tensor(
                        col.type_.as_pygrb,
                        node.table.raw.shape,
                        outdata[cname]
                    )
                else:
                    adddata = from_table[cname].raw.data
                adddata = Tensor(adddata, name=f"INSERT_VALUES.{cname}")

                if node._if_not_exists:
                    # Don't upload stuff that is already in the database
                    adddata = LAIR.Mask(adddata, addstencil)

                if col.is_sideloaded:
                    refcount_sources[cname] = adddata

                column_builders.append(
                    LAIR.Output(
                        ElementwiseAdd(
                            "FIRST",
                            Tensor(col.data, name=f"{node.table.name}.{cname}"),
                            adddata
                        ),
                        f"{node.table.name}.{cname}"
                    )
                )

        #Add the column builder for the output stencil
        column_builders.append(
            LAIR.Output(
                ElementwiseAdd(
                    "FIRST",
                    Tensor(node.table.raw.stencil, name=f"{node.table.name}.stencil"),
                    addstencil
                ),
                f"{node.table.name}.stencil"
            )
        )

        refcount_source_keys = list(refcount_sources.keys())
        refcount_source_values = [LAIR.Output(refcount_sources[k], f"refcounts[{k}]") for k in refcount_source_keys]
        column_builders.extend(refcount_source_values)

        #print("Optimizing LAIR")  ; sys.stdout.flush()
        LAIR.Node.save_graph("IRViz-unoptimized.dot", *column_builders)
        column_builders = [optimize(column) for column in column_builders]
        LAIR.Node.save_graph("IRViz-optimized.dot", *column_builders)
        #print("Running LAIR")  ; sys.stdout.flush()
        column_tensors = list(LAIRVisitor().run(*column_builders))
        #print("Finished executing LAIR") ; sys.stdout.flush()
        
        if len(refcount_sources) > 0:
            column_tensors, refcount_tensors = column_tensors[:-len(refcount_sources)], column_tensors[-len(refcount_sources):]
        else:
            refcount_tensors = []
        out_stencil = column_tensors.pop()
        
        # Ensure the primary key constraints aren't violated by the combination of incoming and existing data
        if (not node._if_not_exists) and (out_stencil.nvals != node.table.raw.stencil.nvals + addstencil.tensor.nvals):
            raise ValueError("Primary key uniqueness constraint violated")

#        for it, (cname, col) in enumerate(node.table.raw.columns):
#          if not col.is_sideloaded:
#            continue
#
#          print("sideloaded column", it, cname, col.data.nvals)
#          if col.data.nvals > 0:
#            continue
#
#          refcount_tensor = refcount_tensors[refcount_source_keys.index(cname)]
#          refcount_values = tensor_to_numpy(refcount_tensor)[-1]
#          result_tensor = column_tensors[it - ndim]
#          result_values = tensor_to_numpy(refcount_tensor)[-1]
#
#          setdiff = set(refcount_values) - set(result_values)
#          print(f"{setdiff=}")
#          assert len(setdiff) == 0


        # Build the reference counts for the output columns
#        print("Building reference counts") ; sys.stdout.flush()
        output_refcounts = {cname: col._refcounts.copy() for cname, col in node.table.raw.columns if col.is_sideloaded}
        assert len(output_refcounts) == len(refcount_source_keys)
        assert len(output_refcounts) == len(refcount_tensors)
        for cname, value_tensor in zip(refcount_source_keys, refcount_tensors):
            values = tensor_to_numpy(value_tensor)[-1]
#            if from_table is not None:
#              print(f"{sys.getrefcount(from_table)=}") ; sys.stdout.flush()
#              print(f"Showing refcounts._print_debug from {from_table.name}.{cname}") ; sys.stdout.flush()
#              from_table[cname].raw._refcounts._print_debug(True)
#              print("Output refcounts for", cname, f"{from_table[cname].raw.data.nvals=}, {values.size=}")
#            else:
#              print("Output refcounts for", cname, f"{nvals=}, {values.size=}")
            output_refcounts[cname].insert_many_encoded_numpy(values)
            #output_refcounts[cname]._print_debug(False)

        #print("Building columns")  ; sys.stdout.flush()
        columns = []
        for it, (cname, col) in enumerate(node.table.raw.columns):
            if col.is_sideloaded:
                refcounts = output_refcounts[cname]
            else:
                refcounts = None

            #print("Instantiating column", cname, col.type_) ; sys.stdout.flush()
            if it >= ndim:
                columns.append(Column(cname, column_tensors[it - ndim], type_=col.type_, refcounts = refcounts))
            else:
                columns.append(Column(cname, None, type_=col.type_, refcounts=refcounts))

            #print("Finished column", cname, col.type_) ; sys.stdout.flush()

        #print("Building table")  ; sys.stdout.flush()
        ret = Table(node.table.raw.shape, columns, stencil_data=out_stencil)
        #print("Replacing table in database")  ; sys.stdout.flush()
        node.db.db[node.table.name] = ret

        return ret
