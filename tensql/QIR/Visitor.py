from typing import Any, List, Dict, Tuple, Sequence, FrozenSet, Optional
import itertools

from .Table import QirTable
from .QirColumn import QirColumn

from .. import QIR, Table, Database
from ..Column import Column
from ..util import SequentialIdGenerator, EncoderDecoder
from .Joiner import Joiner, JoinerResult, JoinMatches

import pygraphblas
import pygraphblas.types
from pygraphblas import Scalar


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
        raise ValueError("Prototype only supports scalars, vectors, and matrices")
    return pattern


class Visitor:
    def visit(self, node: QIR.ABCNode) -> Any:
        for cls in type(node).mro():
            if issubclass(cls, QIR.ABCNode):
                visit_name = f"visit_{cls.__name__}"
                if hasattr(self, visit_name):
                    return getattr(self, visit_name)(node)
        return self.generic_visit(node)

    def generic_visit(self, node: QIR.ABCNode) -> None:
        classes = [cls for cls in type(node).mro() if issubclass(cls, QIR.ABCNode)]
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a visit method for any of {classes}"
        )


class SelectVisitor(Visitor):
    UnaryPyOps = {"-": lambda x: -x, "NOT": lambda x: not x, "ABS": lambda x: abs(x)}

    UnaryGrbOps = {"-": "AINV", "NOT": "LNOT", "ABS": "ABS"}

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

    def __init__(self, join_result: JoinerResult, expected_keys: Sequence[JoinMatches]):
        self._join_result = join_result
        self._expected_keys = tuple(expected_keys)

    @property
    def join_map(self) -> Dict[str, Table]:
        return self._join_result.tables

    def generic_visit(self, node: QIR.ABCNode) -> None:
        classes = [cls for cls in type(node).mro() if issubclass(cls, QIR.ABCNode)]

        raise NotImplementedError(
            f"{type(self).__name__} does not implement a visit method for any of {classes}"
        )

    def visit_QirColumn(self, node: QIR.QirColumn) -> Column:
        return self.join_map[node.table.alias].get_column(node.name).copy()

    def visit_Constant(self, node: QIR.Constant) -> Column:
        ret = Scalar.from_value(node.value)
        return Column(node.name, ret, gbtype=ret.type)

    def visit_ABCUnaryExpression(self, node: QIR.ABCUnaryExpression) -> Column:
        x = self.visit(node.operand)
        if x.is_scalar:
            ret = Scalar.from_value(self.UnaryPyOps[node.op](x.data[0]))
        else:
            ret = x.apply(getattr(x.gbtype, self.UnaryGrbOps[node.op]))
        return Column(node.name, ret, gbtype=ret.type)

    def visit_ArithmeticBinaryExpression(
        self, node: QIR.ArithmeticBinaryExpression
    ) -> Column:
        x = self.visit(node.left)
        y = self.visit(node.right)

        assert x.gbtype == y.gbtype

        grb_op = getattr(x.gbtype, self.BinaryArithmeticGrbOps[node.op])
        if x.is_scalar and y.is_scalar:
            ret = Scalar.from_value(
                self.BinaryArithmeticPyOps[node.op](x.data[0], y.data[0])
            )
        elif (not x.is_scalar) and y.is_scalar:
            ret = x.data.apply_first(y.data, grb_op)
        elif x.is_scalar and (not y.is_scalar):
            ret = x.data.apply_second(y.data, grb_op)
        else:
            ret = x.data.eadd(y.data, grb_op)

        return Column(node.name, ret, gbtype=ret.type)

    def visit_ABCReductionExpression(
        self, node: QIR.ArithmeticBinaryExpression
    ) -> Column:
        x = self.visit(node.arg)

        NODIM = tuple()
        pattern = _get_reduce_pattern(
            self._join_result.primary_keys, self._expected_keys
        )
        grb_op = getattr(x.gbtype, f"{node.op}_MONOID")

        # Scalar Input
        if pattern == (NODIM, NODIM):
            ret = x

        # Vector Input
        elif pattern == ((0,), NODIM):
            ret = x.reduce(grb_op)
        elif pattern == ((0,), (0,)):
            ret = x

        # Matrix Input
        elif pattern == ((0, 1), NODIM):
            ret = x.reduce(grb_op)
        elif pattern == ((0, 1), (0,)):
            ret = x.T.reduce_vector(grb_op)
        elif pattern == ((0, 1), (1,)):
            ret = x.reduce_vector(grb_op)
        elif pattern == ((0, 1), (0, 1)):
            ret = x
        else:
            assert False

        return Column(node.name, ret, gbtype=ret.type)

    def visit_BooleanBinaryExpression(
        self, node: QIR.BooleanBinaryExpression
    ) -> Column:
        x = self.visit(node.left)
        y = self.visit(node.right)

        assert x.gbtype == y.gbtype

        if not x.is_scalar:
            if x.is_vector:
                ret = x.data.__class__.sparse(pygraphblas.types.BOOL, x.data.size)
            else:
                ret = x.data.__class__.sparse(
                    pygraphblas.types.BOOL, x.data.nrows, x.data.ncols
                )
        elif not y.is_scalar:
            if y.is_vector:
                ret = y.data.__class__.sparse(pygraphblas.types.BOOL, y.data.size)
            else:
                ret = y.data.__class__.sparse(
                    pygraphblas.types.BOOL, y.data.nrows, y.data.ncols
                )
        else:
            ret = None

        grb_op = getattr(x.gbtype, self.BinaryBooleanGrbOps[node.op])
        py_op = self.BinaryBooleanPyOps[node.op]
        if x.is_scalar and y.is_scalar:
            ret = Scalar.from_value(py_op(x.data[0], y.data[0]))
        elif (not x.is_scalar) and y.is_scalar:
            x.data.apply_second(grb_op, y.data, out=ret)
        elif x.is_scalar and (not y.is_scalar):
            y.data.apply_first(x.data, grb_op, out=ret)
        else:
            ret = x.data.emult(y.data, grb_op, out=ret)

        return Column(node.name, ret, gbtype=ret.type)


class JoinPlan:
    def __init__(self):
        self.index_constraints = []
        self.primary_keys = []
        self.tables_by_alias = {}

    def add_primary_key(self, pk1: QirColumn) -> None:
        ret = (pk1.table.alias, pk1.name)
        self.primary_keys.append(ret)
        return ret

    def add_constraint(self, pk1: QirColumn, pk2: QirColumn) -> None:
        # assert pk1.is_primary_key
        # assert pk2.is_primary_key
        # assert pk1.primary_key_size == pk2.primary_key_size

        t1 = pk1.table
        t2 = pk2.table.alias
        if self.tables_by_alias.setdefault(pk1.table.alias, pk1.table) is not pk1.table:
            raise ValueError(f"Duplicate table alias detected: {pk1.table.alias!r}")
        if self.tables_by_alias.setdefault(pk2.table.alias, pk2.table) is not pk2.table:
            raise ValueError(f"Duplicate table alias detected: {pk2.table.alias!r}")

        pk1_qname = self.add_primary_key(pk1)
        pk2_qname = self.add_primary_key(pk2)
        self.index_constraints.append((pk1_qname, pk2_qname))

    def add_constraints(
        self, constraints: Sequence[Tuple[QirColumn, QirColumn]]
    ) -> None:
        for pk1, pk2 in constraints:
            self.add_constraint(pk1, pk2)

    def realize(
        self, outkeys: Optional[Sequence[QIR.ArithmeticColumn]]
    ) -> Tuple[JoinMatches, ...]:
        "Groups input primary keys into output primary keys"
        if outkeys is None:
            outkeys = tuple()
        #print("&" * 80)
        #print(f"{outkeys=}")
        #print(f"{self.index_constraints=}")

        # Identify which primary keys need to be merged
        connected_components = []
        matched_pks = set()
        for pk1, pk2 in self.index_constraints:
            pk1_cc, pk2_cc = None, None
            matched_pks.add(pk1)
            matched_pks.add(pk2)
            for it_cc, cc in enumerate(connected_components):
                if pk1 in cc:
                    pk1_cc = it_cc
                if pk2 in cc:
                    pk2_cc = it_cc

            if pk1_cc is None and pk2_cc is None:
                # Create a new connected component
                connected_components.append({pk1, pk2})
            elif pk1_cc is not None and pk2_cc is not None:
                # Merge later connected component into earlier connected component
                if pk1_cc != pk2_cc:
                    target = connected_components[min(pk1_cc, pk2_cc)]
                    later = connected_components.pop(max(pk1_cc, pk2_cc))
                    target |= later
            elif pk1_cc is not None:
                # Add pk2 to pk1's connected component
                connected_components[pk1_cc].add(pk2)
            else:
                # Add pk1 to pk2's connected component
                connected_components[pk2_cc].add(pk1)

        #print("CC1", len(connected_components))
        #print(connected_components)

        # Add primary keys with no joins as stand-alone connected components
        for pk_qname in set(self.primary_keys) - matched_pks:
            connected_components.append([pk_qname])
        #print("CC2", len(connected_components))

        # Re-order the connected components based on outkeys.  Connected components
        # containing an output key should be moved to the front and have the same
        # order as outkeys.
        for outkey in reversed(outkeys):
            outkey_qname = (outkey.table.alias, outkey.name)
            for it_cc, cc in enumerate(connected_components):
                if outkey_qname in cc:
                    connected_components.insert(0, connected_components.pop(it_cc))
                    break
        #print("CC3", len(connected_components))

        return tuple(frozenset(x) for x in connected_components)


class JoinPlanOnClauseVisitor(Visitor):
    def __init__(self, target_table: QirTable):
        self._target_table = target_table

    def visit_Equal(self, node: QIR.Equal) -> List[Tuple[QirColumn, QirColumn]]:
        assert isinstance(node.left, QIR.QirColumn)
        assert isinstance(node.right, QIR.QirColumn)
        # print(node.left.table, node.right.table, self._target_table)
        # print(node.left.table.name, node.right.table.name, self._target_table.name)
        assert self._target_table.name in (node.left.table.name, node.right.table.name)

        assert node.left in node.left.table.primary_keys
        assert node.right in node.right.table.primary_keys
        return [(node.left, node.right)]

    def visit_LogicalAnd(
        self, node: QIR.LogicalAnd
    ) -> List[Tuple[QirColumn, QirColumn]]:
        ret = self.visit(node.left)
        ret.extend(self.visit(node.right))
        return ret


class JoinPlanVisitor(Visitor):
    def visit(self, node: QIR.ABCNode) -> JoinPlan:
        return super().visit(node)

    def visit_From(self, node: QIR.From) -> JoinPlan:
        return self.visit(node.table)

    def visit_QirTable(self, node: QIR.Table) -> JoinPlan:
        ret_map = JoinPlan()
        for pk in node.primary_keys:
            ret_map.add_primary_key(pk)
        return ret_map

    def visit_Join(self, node: QIR.Join) -> JoinPlan:
        ret = self.visit(node.left)
        for pk in node.right.primary_keys:
            ret.add_primary_key(pk)

        if node.on_clause is not None:
            ret.add_constraints(
                JoinPlanOnClauseVisitor(node.right).visit(node.on_clause)
            )
        return ret
