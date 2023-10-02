from __future__ import annotations

from typing import Any, List, Dict, Sequence, Optional
import weakref
import sys
import time

from .. import LAIR
from .. import util
from ..util import GBTensor, grb_shape, ndim, extract_indices, BinaryBooleanGrbOps, build_tensor

import pygraphblas
import pygraphblas.types
from pygraphblas import Scalar, Vector, Matrix
import numpy as np

VisitorReturnType = Optional[GBTensor]

class TempResult:
    def __init__(
        self,
        node: LAIR.Node,
        result: VisitorReturnType,
        children: Sequence[TempResult],
        is_output: bool,
    ):
        self.node = node
        self.result = result
        self.children = tuple(children)
        self.is_output = is_output

class Visitor:
    def visit(self, node: LAIR.Node, *args: Any) -> Any:
        for cls in type(node).mro():
            if issubclass(cls, LAIR.Node):
                visit_name = f"visit_{cls.__name__}"
                if hasattr(self, visit_name):
                    return getattr(self, visit_name)(node, *args)
        return self.generic_visit(node)

    def generic_visit(self, node: LAIR.Node) -> None:
        classes = [cls for cls in type(node).mro() if issubclass(cls, LAIR.Node)]
        if len(classes) > 0:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement a visit method for any of {classes}"
            )
        else:
            raise ValueError(f"Invalid argument: {node!r}")


class ExpressionVisitor(Visitor):
    UnaryPyOps = util.UnaryPyOps
    UnaryGrbOps = util.UnaryGrbOps
    BinaryPyOps = util.BinaryPyOps
    BinaryGrbOps = util.BinaryGrbOps

    def __init__(self, verbose=False) -> None:
        self._expression_cache: Optional[
            Dict[LAIR.Node, TempResult]
        ] = weakref.WeakValueDictionary()
        self.verbose = verbose
        self.recursion = 0

    def _dfs_flatten(self, node, *, is_top_level=True) -> Sequence[TempResult]:
        children = []
        for relationship, child in node:
            children.append(self._dfs_flatten(child, is_top_level=False))
        tmp_result = self._expression_cache.get(node)
        if tmp_result is None:
            tmp_result = TempResult(node, None, children, is_top_level)
            self._expression_cache[node] = tmp_result
        return tmp_result

    def run(self, *nodes: LAIR.Node) -> Sequence[VisitorReturnType]:
        # Perform some reference counting witchcraft with the expression cache to avoid
        # recomputing results, but also avoid wasting memory by storing all temporary results.
        # The expression cache is a weakref.WeakValueDictionary, so Python frees memory of the
        # temporary results when they're no longer in use.  The cache only does anything useful
        # if self._expression_cache has been prepopulated by self.run().

        old_expression_cache_size: int = len(self._expression_cache)

        outputs: List[TempResult] = []
        for node in nodes:
            outputs.append(self._dfs_flatten(node))

        while len(outputs) > 0:
            tmp_result = outputs.pop(0)
            yield self.visit(tmp_result.node)
            del tmp_result

        # Make sure we're not leaking memory
        assert len(self._expression_cache) == old_expression_cache_size

    def visit(self, node: LAIR.Node, *args: Any) -> VisitorReturnType:
        if self.verbose:
            print(" " * self.recursion + "Visiting", node.description)
            sys.stdout.flush()
        self.recursion += 1

        # Perform some reference counting witchcraft with the expression cache to avoid
        # recomputing results, but also avoid wasting memory by storing all temporary results.
        # The expression cache is a weakref.WeakValueDictionary, so Python frees memory of the
        # temporary results when they're no longer in use.  The cache only does anything useful
        # if self._expression_cache has been prepopulated by self.run().
        tmp_result = self._expression_cache.get(node)
        if tmp_result is not None:
            if tmp_result.result is None:
                if self.verbose:
                    print(
                        " " * self.recursion + "Computing result for", node.description
                    )
                start_time = time.time()
                tmp_result.result = super().visit(node, *args)
                stop_time = time.time()
                if self.verbose:
                    print(
                        " " * self.recursion
                        + f"Completed in {stop_time - start_time} seconds",
                        node.description,
                    )

                tmp_result.children = None
            else:
                if self.verbose:
                    print(
                        " " * self.recursion + "Found cached result for",
                        node.description,
                    )
            ret = tmp_result.result
        else:
            if self.verbose:
                print(
                    " " * self.recursion + "Expression cache was missing an entry for",
                    node.description,
                )
            ret = super().visit(node, *args)

        self.recursion -= 1

        if self.verbose:
            print(" " * self.recursion + "Finished", node.description)
            sys.stdout.flush()

        return ret

    def visit_CastLike(self, node: LAIR.CastLike) -> VisitorReturnType:
        x = self.visit(node.operand)
        l = self.visit(node.like)
        if isinstance(x, Scalar):
            ret = Scalar.from_type(l.type)
            if x.nvals > 0:
                ret[None] = x[0]
            return ret
        elif isinstance(x, (Vector, Matrix)):
            return x.cast(l.type)
        else:
            raise ValueError(
                "Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
            )

    def visit_Cast(self, node: LAIR.Cast) -> VisitorReturnType:
        x = self.visit(node.operand)
        if isinstance(x, Scalar):
            ret = Scalar.from_type(node.type_)
            if x.nvals > 0:
                ret[None] = x[0]
            return ret
        elif isinstance(x, Vector):
            ret = Vector.sparse(typ=node.type_, size=x.size)
            identity = getattr(x.type, 'IDENTITY')
            return x.apply(identity, out=ret)
        elif isinstance(x, Matrix):
            ret = Matrix.sparse(node.type_, nrows=x.nrows, ncols=x.ncols)
            identity = getattr(x.type, 'IDENTITY')
            return x.apply(identity, out=ret)
        else:
            raise ValueError(
                "Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
            )

    def visit_DotMxM(self, node: LAIR.DotMxM) -> VisitorReturnType:
        A = self.visit(node.left)
        B = self.visit(node.right)

        assert isinstance(A, Matrix)
        assert isinstance(B, Matrix)

        if (node.add_op, node.mul_op) != ("+", "*"):
            typ: pygraphblas.types.Type = A.type
            add_op = getattr(typ, self.BinaryGrbOps[node.add_op])
            mul_op = getattr(typ, self.BinaryGrbOps[node.mul_op])
            monoid = typ.new_monoid(add_op, typ.default_zero)
            semiring = typ.new_semiring(monoid, mul_op)
            return A.mxm(B, semiring=semiring)
        else:
            return A.mxm(B)

    def visit_DotMxV(self, node: LAIR.DotMxM) -> VisitorReturnType:
        A = self.visit(node.left)
        x = self.visit(node.right)

        assert isinstance(A, Matrix)
        assert isinstance(x, Vector)

        typ: pygraphblas.types.Type = A.type
        add_op = getattr(typ, self.BinaryGrbOps[node.add_op])
        mul_op = getattr(typ, self.BinaryGrbOps[node.mul_op])
        monoid = typ.new_monoid(add_op, typ.default_zero)
        semiring = typ.new_semiring(monoid, mul_op)

        return A.mxv(x, semiring=semiring)

    def visit_DotVxM(self, node: LAIR.DotMxM) -> VisitorReturnType:
        x = self.visit(node.left)
        A = self.visit(node.right)

        assert isinstance(x, Vector)
        assert isinstance(A, Matrix)

        typ: pygraphblas.types.Type = A.type
        add_op = getattr(typ, self.BinaryGrbOps[node.add_op])
        mul_op = getattr(typ, self.BinaryGrbOps[node.mul_op])
        monoid = typ.new_monoid(add_op, typ.default_zero)
        semiring = typ.new_semiring(monoid, mul_op)

        return x.vxm(A, semiring=semiring)

    def visit_ElementwiseAdd(self, node: LAIR.ElementwiseAdd) -> VisitorReturnType:
        x = self.visit(node.left)
        y = self.visit(node.right)

        assert x.type == y.type

        op_name = self.BinaryGrbOps[node.op]
        grb_op = getattr(x.type, op_name)

        if isinstance(x, Scalar) and isinstance(y, Scalar):
            ret = Scalar.from_value(self.BinaryPyOps[node.op](x[0], y[0]))
        elif (not isinstance(x, Scalar)) and isinstance(y, Scalar):
            ret = x.apply_first(y, grb_op)
        elif isinstance(x, Scalar) and (not isinstance(y, Scalar)):
            ret = y.apply_second(y, grb_op)
        elif (
            isinstance(x, (Vector, Matrix))
            and isinstance(y, (Vector, Matrix))
            and ndim(x) == ndim(y)
        ):
            ret = x.eadd(y, grb_op)
        else:
            raise ValueError(
                "Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
            )

        return ret

    def visit_ElementwiseApply(self, node: LAIR.ElementwiseApply) -> VisitorReturnType:
        x = self.visit(node.operand)
        if isinstance(x, Scalar):
            if x.nvals > 0:
                ret = Scalar.from_value(self.UnaryPyOps[node.op](x[0]))
            else:
                ret = Scalar.from_type(x.type)
        elif isinstance(x, (Vector, Matrix)):
            ret = x.apply(getattr(x.type, self.UnaryGrbOps[node.op]))
        else:
            raise ValueError(
                "Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
            )
        return ret

    def visit_ElementwiseApplyBindFirst(
        self, node: LAIR.ElementwiseApplyBindFirst
    ) -> VisitorReturnType:
        x = self.visit(node.operand)
        s = self.visit(node.scalar)

        assert isinstance(s, Scalar)

        if isinstance(x, Scalar):
            if x.nvals > 0:
                ret = Scalar.from_value(self.BinaryPyOps[node.op](s[0], x[0]))
            else:
                ret = Scalar.from_type(x.type)
        elif isinstance(x, (Vector, Matrix)):
            op_name = self.BinaryGrbOps[node.op]
            grb_op = getattr(x.type, op_name)
            ret = x.apply_first(s, grb_op)
        else:
            raise ValueError(
                "Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
            )
        return ret

    def visit_ElementwiseApplyBindSecond(
        self, node: LAIR.ElementwiseApplyBindSecond
    ) -> VisitorReturnType:
        x = self.visit(node.operand)
        s = self.visit(node.scalar)

        assert isinstance(s, Scalar)

        if isinstance(x, Scalar):
            if x.nvals > 0:
                ret = Scalar.from_value(self.BinaryPyOps[node.op](x[0], s[0]))
            else:
                ret = Scalar.from_type(x.type)
        elif isinstance(x, (Vector, Matrix)):
            op_name = self.BinaryGrbOps[node.op]
            grb_op = getattr(x.type, op_name)
            ret = x.apply_second(s, grb_op)
        else:
            raise ValueError(
                "Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
            )
        return ret

    def visit_InnerBroadcastMask(self, node: LAIR.InnerBroadcastMask) -> VisitorReturnType:
        NDIM = tuple()

        x = self.visit(node.left)
        y = self.visit(node.right).nonzero()

        #print(f"{node.pattern=}, {x.type=}, {y.type=}")

        #BOOL = pygraphblas.types.BOOL
        if y.type != x.type:
            if isinstance(y, Scalar):
                ret = Scalar.from_type(x.type)
                if y.nvals > 0:
                    ret[None] = y[0]
            elif isinstance(y, Vector):
                ret = Vector.sparse(typ=x.type, size=y.size)
                identity = getattr(y.type, 'IDENTITY')
                y.apply(identity, out=ret)
            elif isinstance(y, Matrix):
                ret = Matrix.sparse(x.type, nrows=y.nrows, ncols=y.ncols)
                identity = getattr(y.type, 'IDENTITY')
                y.apply(identity, out=ret)
            else:
                raise ValueError(
                    "Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
                )
            y = ret
            del ret
        #print("Executing InnerBroadcastMask", node.pattern)
        #print(f"{node.pattern=}, {x.type=}, {y.type=}")

        op_name = self.BinaryGrbOps[node.op]
        grb_op = getattr(x.type, op_name)

        typ: pygraphblas.types.Type = x.type
        add_op = getattr(typ, "ANY")
        mul_op = getattr(typ, op_name)
        monoid = typ.new_monoid(add_op, typ.default_zero)
        mxm_semiring = typ.new_semiring(monoid, mul_op)
#        monoid.print(level=5)
#        mxm_semiring.print(level=5)
#        print(f"{node.pattern=}, {typ=}, {x.type=}, {y.type=}, {typ.default_zero=}, {op_name=}")

        out_shape = [None for _ in range(len(node.pattern[2]))]
        for it_in, dim_in in enumerate(node.pattern[0]):
            for it_out, dim_out in enumerate(node.pattern[2]):
                if dim_in == dim_out:
                    out_shape[it_out] = grb_shape(x)[it_in]
        for it_in, dim_in in enumerate(node.pattern[1]):
            for it_out, dim_out in enumerate(node.pattern[2]):
                if dim_in == dim_out:
                    out_shape[it_out] = grb_shape(y)[it_in]

        assert None not in out_shape

        ret = build_tensor(x.type, out_shape)

        if x.nvals == 0 or y.nvals == 0:
            pass

        elif node.pattern == (NDIM, NDIM, NDIM):
            ret[None] = self.BinaryPyOps[op_name](x[0], y[0])

        elif node.pattern == (NDIM, (0,), (0,)):
            y.apply_first(x, grb_op, out=ret)
        elif node.pattern == (NDIM, (0,1), (0,1)):
            y.apply_first(x, grb_op, out=ret)
        elif node.pattern == (NDIM, (1,0), (0,1)):
            y.apply_first(x, grb_op, out=ret)
            ret = ret.T

        elif node.pattern == ((0,), NDIM, (0,)):
            x.apply_second(y, grb_op, out=ret)
        elif node.pattern == ((0,1), NDIM, (0,1)):
            x.apply_second(y, grb_op, out=ret)
        elif node.pattern == ((1,0), NDIM, (0,1)):
            x.apply_second(y, grb_op, out=ret)
            ret = ret.T

        elif node.pattern == ((0,), (0,), (0,)):
            x.emult(y, mult_op=grb_op, out=ret)
        elif node.pattern == ((0,), (1,), (0, 1)):
            mx = Matrix.sparse(x.type, x.size, 1)
            mx.assign_col(0, x)
            my = Matrix.sparse(y.type, 1, y.size)
            my.assign_row(0, y)
            mx.mxm(my, semiring=mxm_semiring, out=ret)
        elif node.pattern == ((1,), (0,), (0, 1)):
            mx = Matrix.sparse(x.type, x.size, 1)
            mx.assign_col(0, x)
            my = Matrix.sparse(y.type, 1, y.size)
            my.assign_row(0, y)
            mx.mxm(my, semiring=mxm_semiring, out=ret)
            ret = ret.T

        elif node.pattern == ((0,), (0, 1), (0, 1)):
            mx = Matrix.from_diag(x)
            mx.mxm(y, semiring=mxm_semiring, out=ret)
        elif node.pattern == ((1,), (0, 1), (0, 1)):
            mx = Matrix.from_diag(x)
            mx.mxm(y.T, semiring=mxm_semiring, out=ret)
            ret = ret.T
        elif node.pattern == ((0,), (1, 0), (0, 1)):
            mx = Matrix.from_diag(x)
            mx.mxm(y.T, semiring=mxm_semiring, out=ret)
        elif node.pattern == ((1,), (1, 0), (0, 1)):
            mx = Matrix.from_diag(x)
            mx.mxm(y, semiring=mxm_semiring, out=ret)
            ret = ret.T

        elif node.pattern == ((0, 1), (0,), (0, 1)):
            my = Matrix.from_diag(y)
            x.T.mxm(my, semiring=mxm_semiring, out=ret)
            ret = ret.T
        elif node.pattern == ((0, 1), (1,), (0, 1)):
            my = Matrix.from_diag(y)
            x.mxm(my, semiring=mxm_semiring, out=ret)
        elif node.pattern == ((1, 0), (0,), (0, 1)):
            my = Matrix.from_diag(y)
            x.mxm(my, semiring=mxm_semiring, out=ret)
            ret = ret.T
        elif node.pattern == ((1, 0), (1,), (0, 1)):
            my = Matrix.from_diag(y)
            x.T.mxm(my, semiring=mxm_semiring, out=ret)

        elif node.pattern == ((0,1), (0,1), (0,1)):
            x.emult(y, mult_op=grb_op, out=ret)
        elif node.pattern == ((0,1), (1,0), (0,1)):
            x.emult(y.T, mult_op=grb_op, out=ret)
        elif node.pattern == ((1,0), (0,1), (0,1)):
            x.T.emult(y, mult_op=grb_op, out=ret)
        elif node.pattern == ((1,0), (1,0), (0,1)):
            x.emult(y, mult_op=grb_op, out=ret)
            ret = ret.T

        else:
            print(node.pattern)
            raise ValueError(
                f"Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
            )

        #print(f"{node.pattern=},{ret.type=},{x.nvals=},{y.nvals=},{ret.nvals=}")
        
        return ret

    def visit_InnerBroadcast(self, node: LAIR.InnerBroadcast) -> VisitorReturnType:
        NDIM = tuple()

        x = self.visit(node.left)
        y = self.visit(node.right)

#        if x.type != y.type:
#            if isinstance(y, (Vector, Matrix)):
#                y = y.cast(x.type)
#            elif isinstance(y, Scalar):
#                new_y = Scalar.from_type(x.type)
#                new_y[None] = y[None]
#                y = new_y
#            else:
#                raise ValueError(
#                    "Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
#                )
        if x.type != y.type:
            if isinstance(y, Scalar):
                ret = Scalar.from_type(x.type)
                if y.nvals > 0:
                    ret[None] = y[0]
            elif isinstance(y, Vector):
                ret = Vector.sparse(typ=x.type, size=y.size)
                identity = getattr(y.type, 'IDENTITY')
                y.apply(identity, out=ret)
            elif isinstance(y, Matrix):
                ret = Matrix.sparse(x.type, nrows=y.nrows, ncols=y.ncols)
                identity = getattr(y.type, 'IDENTITY')
                y.apply(identity, out=ret)
            else:
                raise ValueError(
                    "Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
                )
            y = ret

        op_name = self.BinaryGrbOps[node.op]

        typ: pygraphblas.types.Type = x.type
        grb_op = getattr(typ, op_name)
        is_boolean = node.op in util.BinaryBooleanGrbOps
        if is_boolean:
          add_op = getattr(pygraphblas.types.BOOL, "ANY")
          mul_op = grb_op
          zero = False
        else:
          add_op = getattr(typ, "ANY")
          mul_op = grb_op
          zero = typ.default_zero
        #print(f"{typ=},ANY,{add_op.get_op()=},{op_name},{mul_op.get_op()=},{zero=},{is_boolean=}")
        monoid = typ.new_monoid(add_op, zero)
        mxm_semiring = typ.new_semiring(monoid, mul_op)

        out_shape = [None for _ in range(len(node.pattern[2]))]
        for it_in, dim_in in enumerate(node.pattern[0]):
            for it_out, dim_out in enumerate(node.pattern[2]):
                if dim_in == dim_out:
                    out_shape[it_out] = grb_shape(x)[it_in]
        for it_in, dim_in in enumerate(node.pattern[1]):
            for it_out, dim_out in enumerate(node.pattern[2]):
                if dim_in == dim_out:
                    out_shape[it_out] = grb_shape(y)[it_in]

        assert None not in out_shape

        #print("Executing InnerBroadcast", node.pattern, mxm_semiring, f"{mxm_semiring.pls=}", f"{mxm_semiring.mul=}", f"{mxm_semiring.name=}")

        if x.nvals == 0 or y.nvals == 0:
            if len(node.pattern[2]) == 0:
                ret = Scalar.from_type(x.type)
            elif len(node.pattern[2]) == 1:
                ret = Vector.sparse(x.type, *out_shape)
            elif len(node.pattern[2]) == 2:
                ret = Matrix.sparse(x.type, *out_shape)
            else:
                raise ValueError(
                    f"Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
                )

        elif node.pattern == (NDIM, NDIM, NDIM):
            ret = Scalar.from_value(self.BinaryPyOps[op_name](x[0], y[0]))

        elif node.pattern == (NDIM, (0,), (0,)):
            ret = y.apply_first(first=x, op=grb_op)
        elif node.pattern == (NDIM, (0,1), (0,1)):
            ret = y.apply_first(first=x, op=grb_op)
        elif node.pattern == (NDIM, (1,0), (0,1)):
            ret = y.apply_first(first=x, op=grb_op).T

        elif node.pattern == ((0,), NDIM, (0,)):
            ret = x.apply_second(second=y, op=grb_op)
        elif node.pattern == ((0,1), NDIM, (0,1)):
            ret = x.apply_second(second=y, op=grb_op)
        elif node.pattern == ((1,0), NDIM, (0,1)):
            ret = x.apply_second(second=y, op=grb_op).T

        elif node.pattern == ((0,), (0,), (0,)):
            ret = x.emult(y, mult_op=grb_op)
        elif node.pattern == ((0,), (1,), (0, 1)):
            mx = Matrix.sparse(x.type, x.size, 1)
            mx.assign_col(0, x)
            my = Matrix.sparse(y.type, 1, y.size)
            my.assign_row(0, y)
            ret = mx.mxm(my, semiring=mxm_semiring)
        elif node.pattern == ((1,), (0,), (0, 1)):
            mx = Matrix.sparse(x.type, x.size, 1)
            mx.assign_col(0, x)
            my = Matrix.sparse(y.type, 1, y.size)
            my.assign_row(0, y)
            ret = mx.mxm(my, semiring=mxm_semiring).T

        elif node.pattern == ((0,), (0, 1), (0, 1)):
            mx = Matrix.from_diag(x)
            ret = mx.mxm(y, semiring=mxm_semiring)
        elif node.pattern == ((1,), (0, 1), (0, 1)):
            mx = Matrix.from_diag(x)
            ret = mx.mxm(y.T, semiring=mxm_semiring).T
        elif node.pattern == ((0,), (1, 0), (0, 1)):
            mx = Matrix.from_diag(x)
            ret = mx.mxm(y.T, semiring=mxm_semiring)
        elif node.pattern == ((1,), (1, 0), (0, 1)):
            mx = Matrix.from_diag(x)
            ret = mx.mxm(y, semiring=mxm_semiring).T

        elif node.pattern == ((0, 1), (0,), (0, 1)):
            my = Matrix.from_diag(y)
            ret = x.T.mxm(my, semiring=mxm_semiring).T
            #print("&" * 80)
            #print(f"{x.nvals=},{my.nvals=},{ret.nvals=},{x.nonzero().nvals=},{my.nonzero().nvals=}")
            #print("&" * 80)
        elif node.pattern == ((0, 1), (1,), (0, 1)):
            my = Matrix.from_diag(y)
            ret = x.mxm(my, semiring=mxm_semiring)
        elif node.pattern == ((1, 0), (0,), (0, 1)):
            my = Matrix.from_diag(y)
            ret = x.mxm(my, semiring=mxm_semiring).T
        elif node.pattern == ((1, 0), (1,), (0, 1)):
            my = Matrix.from_diag(y)
            ret = x.T.mxm(my, semiring=mxm_semiring)

        elif node.pattern == ((0,1), (0,1), (0,1)):
            ret = x.emult(y, mult_op=grb_op)
        elif node.pattern == ((0,1), (1,0), (0,1)):
            ret = x.emult(y.T, mult_op=grb_op)
        elif node.pattern == ((1,0), (0,1), (0,1)):
            ret = x.T.emult(y, mult_op=grb_op)
        elif node.pattern == ((1,0), (1,0), (0,1)):
            ret = x.emult(y, mult_op=grb_op).T

        else:
            print(node.pattern)
            raise ValueError(
                f"Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
            )

        #print(f"InnerBroadcast,{node.pattern=},{x.nvals=},{y.nvals=},{ret.nvals=}")
        
        return ret


    def visit_ElementwiseMultiply(
        self, node: LAIR.ElementwiseMultiply
    ) -> VisitorReturnType:
        x = self.visit(node.left)
        y = self.visit(node.right)

        if x.type != y.type:
            if isinstance(y, (Vector, Matrix)):
                y = y.cast(x.type)
            elif isinstance(y, Scalar):
                new_y = Scalar.from_type(x.type)
                new_y[None] = y[None]
                y = new_y
            else:
                raise ValueError(
                    "Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
                )

        op_name = self.BinaryGrbOps[node.op]

        grb_op = getattr(x.type, op_name)

        if isinstance(x, Scalar) and isinstance(y, Scalar):
            if x.nvals > 0 and y.nvals > 0:
                ret = Scalar.from_value(self.BinaryPyOps[node.op](x[0], y[0]))
            else:
                ret = Scalar.from_type(x.type)
        elif (not isinstance(x, Scalar)) and isinstance(y, Scalar):
            if x.nvals > 0 and y.nvals > 0:
                ret = x.apply_second(grb_op, y)
            elif isinstance(x, Vector):
                ret = Vector.sparse(x.type, x.size)
            elif isinstance(x, Matrix):
                ret = Matrix.sparse(x.type, x.nrows, x.ncols)
        elif isinstance(x, Scalar) and (not isinstance(y, Scalar)):
            if x.nvals > 0 and y.nvals > 0:
                ret = y.apply_first(x, grb_op)
            elif isinstance(y, Vector):
                ret = Vector.sparse(y.type, y.size)
            elif isinstance(y, Matrix):
                ret = Matrix.sparse(y.type, y.nrows, y.ncols)
        elif (
            isinstance(x, (Vector, Matrix))
            and isinstance(y, (Vector, Matrix))
            and ndim(x) == ndim(y)
        ):
            ret = x.emult(y, grb_op)
        elif (
            isinstance(x, (Scalar, Vector, Matrix))
            and isinstance(y, (Scalar, Vector, Matrix))
            and ndim(x) != ndim(y)
        ):
            raise ValueError(f"Ndim of x and y must be identical")
        else:
            raise ValueError(
                f"Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
            )

        return ret

    def visit_InnerBroadcastStencil(
        self, node: LAIR.InnerBroadcastStencil
    ) -> VisitorReturnType:
        raise ValueError(
            f"The requested promotion operation is not supported by GraphBLAS {node.left_indices}, {node.right_indices} -> {node.out_indices}"
        )

    def visit_AntiMask(self, node: LAIR.AntiMask) -> VisitorReturnType:
        operand = self.visit(node.operand)
        mask = self.visit(node.mask)

        assert mask.type == pygraphblas.types.BOOL
        assert grb_shape(operand) == grb_shape(mask)

        if isinstance(operand, Scalar) and isinstance(mask, Scalar):
            if operand.nvals > 0 and mask.nvals == 0:
                ret = operand
            else:
                ret = Scalar.from_type(operand.type)
        elif isinstance(operand, Vector) and isinstance(mask, Vector):
            ret = operand.extract(mask, desc=pygraphblas.descriptor.C)
        elif isinstance(operand, Matrix) and isinstance(mask, Matrix):
            ret = operand.extract_matrix(mask=mask, desc=pygraphblas.descriptor.C)
        elif (
            isinstance(operand, (Scalar, Vector, Matrix))
            and isinstance(mask, (Scalar, Vector, Matrix))
            and ndim(operand) != ndim(mask)
        ):
            raise ValueError(f"Ndim of x and y must be identical")
        else:
            raise ValueError(
                f"Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
            )

        return ret

    def visit_Mask(self, node: LAIR.Mask) -> VisitorReturnType:
        operand = self.visit(node.operand)
        mask = self.visit(node.mask)

        assert mask.type == pygraphblas.types.BOOL
        assert ndim(operand) == ndim(mask)

        if isinstance(operand, Scalar) and isinstance(mask, Scalar):
            if operand.nvals > 0 and mask.nvals > 0:
                ret = operand
            else:
                ret = Scalar.from_type(operand.type)
        elif isinstance(operand, Vector) and isinstance(mask, Vector):
            ret = operand.extract(mask)
        elif isinstance(operand, Matrix) and isinstance(mask, Matrix):
            ret = operand.extract_matrix(mask=mask)
        elif (
            isinstance(operand, (Scalar, Vector, Matrix))
            and isinstance(mask, (Scalar, Vector, Matrix))
            and ndim(operand) != ndim(mask)
        ):
            raise ValueError(f"Ndim of x and y must be identical")
        else:
            raise ValueError(
                f"Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
            )

        return ret

    def visit_Output(self, node: LAIR.Output) -> VisitorReturnType:
        return self.visit(node.operand)

    def visit_Pattern(self, node: LAIR.Pattern) -> VisitorReturnType:
        x = self.visit(node.operand)

        if isinstance(x, Scalar):
            if x.nvals > 0 and float(x[0]) != 0.0:
                ret = Scalar.from_value(True)
            else:
                ret = Scalar.from_type(x.type)
        elif isinstance(x, Vector):
            ret = x.nonzero().pattern()
        elif isinstance(x, Matrix):
            ret = x.nonzero().pattern()
        else:
            raise ValueError(
                "Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
            )

        # print(node.description,"result", ret.nvals)

        return ret

    def visit_PrimaryKey(self, node: LAIR.PrimaryKey) -> VisitorReturnType:
        operand = self.visit(node.stencil)
        return extract_indices(operand, axis=node.colnum, typ=node.type_)

    def visit_Promote(self, node: LAIR.Promote) -> VisitorReturnType:
#        if node._operand_indices == node._out_indices:
#            return node
#
#        assert all(idx in self._out_indices for idx in self._operand_indices)
#
#        sig = node.signature
#        NDIM = tuple()
#        if sig == (NDIM, (0,)):
#
#        elif sig == (NDIM, (0, 1)):
#        elif sig == ((0,), (0, 1)):
#        elif sig == ((0,), (1, 0)):
#        else:
#            raise ValueError(
#                "Prototype only supports PyGraphBLAS scalars, vectors, and matrices"
#            )
#
#        ndim_in = len(node._operand_indices)
#        ndim_out = len(node._out_indices)
#        if ndim_in > ndim_out:
#            raise
        raise ValueError(
            f"The requested promotion operation is not supported by GraphBLAS {node.operand_indices} -> {node.out_indices}"
        )

    def visit_Reduction(self, node: LAIR.Reduction) -> VisitorReturnType:
        operand = self.visit(node.operand)

        NODIM = tuple()

        orig_axes = tuple(range(ndim(operand)))
        pattern = (orig_axes, tuple(ax for ax in orig_axes if ax not in node.axes))
        if 0:
            #Doesn't work, see: https://github.com/Graphegon/pygraphblas/issues/60
            grb_op = getattr(operand.type, f"{self.BinaryGrbOps[node.op]}_MONOID")
        else:
            add_op = getattr(operand.type, self.BinaryGrbOps[node.op])
            grb_op = operand.type.new_monoid(add_op, operand.type.default_zero)

        # Scalar Input
        if isinstance(operand, Scalar) and pattern == (NODIM, NODIM):
            ret = operand

        # Vector Input
        elif isinstance(operand, Vector) and pattern == ((0,), NODIM):
            ret = Scalar.from_type(operand.type)
            if operand.nvals > 0:
                ret[None] = operand.reduce(grb_op)
        elif isinstance(operand, Vector) and pattern == ((0,), (0,)):
            ret = operand

        # Matrix Input
        elif isinstance(operand, Matrix) and pattern == ((0, 1), NODIM):
            ret = Scalar.from_type(operand.type)
            if operand.nvals > 0:
                retval = operand.T.reduce(grb_op)
                ret[None] = retval
        elif isinstance(operand, Matrix) and pattern == ((0, 1), (0,)):
            ret = operand.reduce_vector(grb_op)
        elif isinstance(operand, Matrix) and pattern == ((0, 1), (1,)):
            ret = operand.T.reduce_vector(grb_op)
        elif isinstance(operand, Matrix) and pattern == ((0, 1), (0, 1)):
            ret = operand

        else:
            raise ValueError(
                f"Invalid reduction pattern {pattern[0]}->{pattern[1]} for operand with type={type(operand)}"
            )

        return ret

    def visit_Tensor(self, node: LAIR.Tensor) -> VisitorReturnType:
        return node.tensor

    def visit_PermuteIndices(self, node: LAIR.PermuteIndices) -> VisitorReturnType:
        operand = self.visit(node.operand)
        axes = node.new_axes or tuple(reversed(range(ndim(operand))))

        if isinstance(operand, Scalar):
            ret = operand
        elif isinstance(operand, Vector):
            ret = operand
        elif isinstance(operand, Matrix) and axes == (0, 1):
            ret = operand
        elif isinstance(operand, Matrix) and axes == (1, 0):
            ret = operand.T
        else:
            raise ValueError(
                f"Unknown transpose type with new_axes={axes} and operand type {type(operand)}"
            )

        return ret

    def visit_VectorToColumnMatrix(
        self, node: LAIR.VectorToColumnMatrix
    ) -> VisitorReturnType:
        arg = self.visit(node.operand)
        assert isinstance(arg, Vector)
        ret: Matrix = Matrix.sparse(typ=arg.type, nrows=arg.size, ncols=1)
        ret.assign_col(0, arg)
        return ret

    def visit_VectorToRowMatrix(
        self, node: LAIR.VectorToRowMatrix
    ) -> VisitorReturnType:
        arg = self.visit(node.operand)
        assert isinstance(arg, Vector)
        ret: Matrix = Matrix.sparse(typ=arg.type, ncols=arg.size, nrows=1)
        ret.assign_row(0, arg)
        return ret

    def visit_VectorToDiagMatrix(
        self, node: LAIR.VectorToDiagMatrix
    ) -> VisitorReturnType:
        arg = self.visit(node.operand)
        assert isinstance(arg, Vector)
        ret: Matrix = Matrix.from_diag(arg)
        return ret
