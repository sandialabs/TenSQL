from typing import Optional, Tuple, Set, FrozenSet, List, Dict
import itertools

from ...util import GBTensor, ndim, grb_shape

from ..Visitor import Visitor

from ... import LAIR

VisitorReturnType = Tuple[Optional[int], ...]

class ShapeVisitor(Visitor):
    def __init__(self):
        self.registry = {}

    def visit(self, node: LAIR.Node) -> VisitorReturnType:
        ret = super().visit(node)
        self.registry[node] = ret
        return ret

    def visit_InnerBroadcast(self, node: LAIR.InnerBroadcast) -> VisitorReturnType:
        dim_sizes = {}

        left_shape = self.visit(node.left)
        for idx, size in zip(node.left_indices, left_shape):
            assert idx not in dim_sizes
            dim_sizes[idx] = size

        right_shape = self.visit(node.right)
        for idx, size in zip(node.left_indices, left_shape):
            if idx in dim_sizes:
                assert dim_sizes[idx] == size
            else:
                dim_sizes[idx] = size

        return tuple(dim_sizes[oidx] for oidx in node.out_indices)

    def visit_InnerBroadcastMask(self, node: LAIR.InnerBroadcastMask) -> VisitorReturnType:
        dim_sizes = {}

        left_shape = self.visit(node.left)
        for idx, size in zip(node.left_indices, left_shape):
            assert idx not in dim_sizes
            dim_sizes[idx] = size

        right_shape = self.visit(node.right)
        for idx, size in zip(node.left_indices, left_shape):
            if idx in dim_sizes:
                assert dim_sizes[idx] == size
            else:
                dim_sizes[idx] = size

        return tuple(dim_sizes[oidx] for oidx in node.out_indices)

    def visit_CastLike(self, node: LAIR.CastLike) -> VisitorReturnType:
        return self.visit(node.operand)

    def visit_DotMxM(self, node: LAIR.DotMxM) -> VisitorReturnType:
        left_shape = self.visit(node.left)
        right_shape = self.visit(node.right)
        return [left_shape[0], right_shape[1]]

    def visit_DotMxV(self, node: LAIR.DotMxV) -> VisitorReturnType:
        left_shape = self.visit(node.left)
        self.visit(node.right)
        return (left_shape[0],)

    def visit_DotVxM(self, node: LAIR.DotVxM) -> VisitorReturnType:
        self.visit(node.left)
        right_shape = self.visit(node.right)
        return (right_shape[0],)

    def visit_ElementwiseAdd(self, node: LAIR.ElementwiseAdd) -> VisitorReturnType:
        left_shape = self.visit(node.left)
        right_shape = self.visit(node.right)
        assert len(left_shape) == len(right_shape)
        return left_shape

    def visit_ElementwiseApply(self, node: LAIR.ElementwiseApply) -> VisitorReturnType:
        return self.visit(node.operand)

    def visit_Cast(self, node: LAIR.Cast) -> VisitorReturnType:
        return self.visit(node.operand)

    def visit_ElementwiseApplyBindFirst(
        self, node: LAIR.ElementwiseApplyBindFirst
    ) -> VisitorReturnType:
        return self.visit(node.operand)

    def visit_ElementwiseApplyBindSecond(
        self, node: LAIR.ElementwiseApplyBindSecond
    ) -> VisitorReturnType:
        return self.visit(node.operand)

    def visit_ElementwiseMultiply(
        self, node: LAIR.ElementwiseMultiply
    ) -> VisitorReturnType:
        left_shape = self.visit(node.left)
        right_shape = self.visit(node.right)
        assert len(left_shape) == len(right_shape)
        return left_shape

    def visit_AntiMask(self, node: LAIR.AntiMask) -> VisitorReturnType:
        operand_shape = self.visit(node.operand)
        mask_shape = self.visit(node.mask)
        assert len(operand_shape) == len(mask_shape)
        return operand_shape

    def visit_Mask(self, node: LAIR.Mask) -> VisitorReturnType:
        operand_shape = self.visit(node.operand)
        mask_shape = self.visit(node.mask)
        assert len(operand_shape) == len(mask_shape)
        return operand_shape

    def visit_Output(self, node: LAIR.Output) -> VisitorReturnType:
        return self.visit(node.operand)

    def visit_Pattern(self, node: LAIR.Pattern) -> VisitorReturnType:
        return self.visit(node.operand)

    # def visit_PrimaryKey(self, node: LAIR.PrimaryKey) -> VisitorReturnType:
    #    return self.visit(node.)

    def visit_Promote(self, node: LAIR.Promote) -> VisitorReturnType:
        ret = []
        operand_shape = self.visit(node.operand)
        for oidx in node.out_indices:
            if oidx in node.operand_indices:
                ret.append(operand_shape[oidx])
            else:
                ret.append(None)
        return tuple(ret)

    def visit_Reduction(self, node: LAIR.Reduction) -> VisitorReturnType:
        operand_shape = self.visit(node.operand)
        return tuple(
            oidx
            for it_oidx, oidx in enumerate(operand_shape)
            if it_oidx not in node.axes
        )

    def visit_Tensor(self, node: LAIR.Tensor) -> VisitorReturnType:
        ret = grb_shape(node.tensor)
        if node.stencil is not None:
            assert grb_shape(node.stencil) == ret
        return tuple(ret)

    def visit_PermuteIndices(self, node: LAIR.PermuteIndices) -> VisitorReturnType:
        operand_shape = self.visit(node.operand)
        if node.new_axes is None:
            return tuple(reversed(operand_shape))
        else:
            assert set(node.new_axes) == set(range(len(operand_shape)))
            return tuple(operand_shape[dim] for dim in node.new_axes)

    def visit_VectorToColumnMatrix(self, node: LAIR.VectorToColumnMatrix):
        operand_shape = self.visit(node.operand)
        return (operand_shape[0], 1)

    def visit_VectorToRowMatrix(self, node: LAIR.VectorToRowMatrix):
        operand_shape = self.visit(node.operand)
        return (1, operand_shape[0])

    def visit_VectorToDiagMatrix(self, node: LAIR.VectorToDiagMatrix):
        operand_shape = self.visit(node.operand)
        return (operand_shape[0], operand_shape[0])
