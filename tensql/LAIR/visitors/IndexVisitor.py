from typing import Optional, Tuple, Set, FrozenSet, List, Dict

from ...util import GBTensor, ndim
from ..Visitor import Visitor
from ... import LAIR
from .index_algebra import intersect_sop, union_sop, complement_sop, SOPExpression

VisitorReturnType = List[SOPExpression]

class IndexVisitor(Visitor):
    def __init__(self):
        self.registry = {}

    def visit(self, node: LAIR.Node) -> VisitorReturnType:
        ret = super().visit(node)
        self.registry[node] = ret
        return ret

    def visit_InnerBroadcastMask(self, node: LAIR.InnerBroadcastMask) -> VisitorReturnType:
        left_indices = self.visit(node.left)
        right_indices = self.visit(node.right)

        ret = [[] for it_out in node.pattern[2]]
        for it_in, dim_in in enumerate(node.pattern[0]):
            for it_out, dim_out in enumerate(node.pattern[2]):
                if dim_in == dim_out:
                    ret[it_out].append(left_indices[it_in])
        for it_in, dim_in in enumerate(node.pattern[1]):
            for it_out, dim_out in enumerate(node.pattern[2]):
                if dim_in == dim_out:
                    ret[it_out].append(right_indices[it_in])
        return [intersect_sop(*idxset) for idxset in ret]

    def visit_InnerBroadcast(self, node: LAIR.InnerBroadcast) -> VisitorReturnType:
        left_indices = self.visit(node.left)
        right_indices = self.visit(node.right)

        ret = [[] for it_out in node.pattern[2]]
        for it_in, dim_in in enumerate(node.pattern[0]):
            for it_out, dim_out in enumerate(node.pattern[2]):
                if dim_in == dim_out:
                    ret[it_out].append(left_indices[it_in])
        for it_in, dim_in in enumerate(node.pattern[1]):
            for it_out, dim_out in enumerate(node.pattern[2]):
                if dim_in == dim_out:
                    ret[it_out].append(right_indices[it_in])
        return [intersect_sop(*idxset) for idxset in ret]

    def visit_CastLike(self, node: LAIR.CastLike) -> VisitorReturnType:
        return self.visit(node.operand)

    def visit_DotMxM(self, node: LAIR.DotMxM) -> VisitorReturnType:
        left_indices = self.visit(node.left)
        right_indices = self.visit(node.right)
        return [left_indices[0], right_indices[1]]

    def visit_DotMxV(self, node: LAIR.DotMxV) -> VisitorReturnType:
        left_indices = self.visit(node.left)
        right_indices = self.visit(node.right)
        return [
            left_indices[0],
            #intersect_sop(left_indices[1], right_indices[0]),
        ]

    def visit_DotVxM(self, node: LAIR.DotVxM) -> VisitorReturnType:
        left_indices = self.visit(node.left)
        right_indices = self.visit(node.right)
        return [
            #intersect_sop(left_indices[0], right_indices[0]),
            right_indices[1],
        ]

    def visit_ElementwiseAdd(self, node: LAIR.ElementwiseAdd) -> VisitorReturnType:
        left_indices = self.visit(node.left)
        right_indices = self.visit(node.right)
        assert len(left_indices) == len(right_indices)
        return [
            union_sop(lidx, ridx)
            for lidx, ridx in zip(left_indices, right_indices)
        ]

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
        left_indices = self.visit(node.left)
        right_indices = self.visit(node.right)
        # print(len(left_indices), len(right_indices))
        assert (len(left_indices) == len(right_indices) or min(len(left_indices), len(right_indices)) == 0)
        if len(left_indices) == len(right_indices):
            out_indices = [
                intersect_sop(lidx, ridx)
                for lidx, ridx in zip(left_indices, right_indices)
            ]
        elif len(left_indices) == 0:
            out_indices = right_indices
        elif len(right_indices) == 0:
            out_indices = left_indices
        # print("IndexVisitor.visit_ElementwiseMultiply", out_indices, left_indices, right_indices)
        return out_indices

    def visit_AntiMask(self, node: LAIR.AntiMask) -> VisitorReturnType:
        operand_indices = self.visit(node.operand)
        mask_indices = self.visit(node.mask)
        assert len(operand_indices) == len(mask_indices)
        return [
            intersect_sop(oidx, complement_sop(midx))
            for oidx, midx in zip(operand_indices, mask_indices)
        ]

    def visit_Mask(self, node: LAIR.Mask) -> VisitorReturnType:
        operand_indices = self.visit(node.operand)
        mask_indices = self.visit(node.mask)
        assert len(operand_indices) == len(mask_indices)
        return [
            intersect_sop(oidx, midx)
            for oidx, midx in zip(operand_indices, mask_indices)
        ]

    def visit_Output(self, node: LAIR.Output) -> VisitorReturnType:
        return self.visit(node.operand)

    def visit_Pattern(self, node: LAIR.Pattern) -> VisitorReturnType:
        operand_indices = self.visit(node.operand)
        
        return [
            intersect_sop(oidx, frozenset({frozenset({(id(node), it, False)})}))
            for it, oidx in enumerate(operand_indices)
        ]

    def visit_PrimaryKey(self, node: LAIR.PrimaryKey) -> VisitorReturnType:
        return self.visit(node.stencil)

    def visit_Promote(self, node: LAIR.Promote) -> VisitorReturnType:
        ret = []
        operand_indices = self.visit(node.operand)
        for oidx in node.out_indices:
            if oidx in node.operand_indices:
                ret.append(operand_indices[oidx])
            else:
                ret.append(frozenset())
        return ret

    def visit_Reduction(self, node: LAIR.Reduction) -> VisitorReturnType:
        operand_indices = self.visit(node.operand)
        return [
            oidx
            for it_oidx, oidx in enumerate(operand_indices)
            if it_oidx not in node.axes
        ]

    def visit_Tensor(self, node: LAIR.Tensor) -> VisitorReturnType:
        if node.stencil is None:
            return [
                frozenset({frozenset({(id(node.tensor), it, False)})})
                for it in range(ndim(node.tensor))
            ]
        else:
            stencil_indices = self.visit(node.stencil)
            return [
                intersect_sop(
                    frozenset({frozenset({(id(node.tensor), it, False)})}), stencil_indices[it]
                )
                for it in range(ndim(node.tensor))
            ]

    def visit_PermuteIndices(self, node: LAIR.PermuteIndices) -> VisitorReturnType:
        operand_indices = self.visit(node.operand)
        if node.new_axes is None:
            return list(reversed(operand_indices))
        else:
            assert set(node.new_axes) == set(range(len(operand_indices)))
            return [operand_indices[dim] for dim in node.new_axes]

    def visit_VectorToColumnMatrix(self, node: LAIR.VectorToColumnMatrix):
        operand_indices = self.visit(node.operand)
        return [operand_indices[0], [frozenset({frozenset({(None, 0, False)})})]]

    def visit_VectorToRowMatrix(self, node: LAIR.VectorToRowMatrix):
        operand_indices = self.visit(node.operand)
        return [[frozenset({frozenset({(None, 0, False)})})], operand_indices[0]]

    def visit_VectorToDiagMatrix(self, node: LAIR.VectorToDiagMatrix):
        operand_indices = self.visit(node.operand)
        return [operand_indices[0], operand_indices[0]]
