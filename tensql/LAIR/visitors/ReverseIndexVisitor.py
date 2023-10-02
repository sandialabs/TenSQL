from typing import Optional, Tuple, Set, FrozenSet, List, Dict
import copy

from ...util import GBTensor, ndim
from ..Visitor import Visitor
from ... import LAIR
from .index_algebra import intersect_sop, union_sop, complement_sop, SOPExpression


Memo = Tuple[SOPExpression]
VisitorReturnType = Memo

class ReverseIndexVisitor(Visitor):
    def __init__(self, forward_registry: Dict[LAIR.Node, List[SOPExpression]]):
        self.forward_registry = {
            k: copy.deepcopy(v) for k, v in forward_registry.items()
        }
        self.registry: Dict[LAIR.Node, List[SOPExpression]] = {}

    def visit(
        self, node: LAIR.Node, out_indices: Optional[Memo] = None
    ) -> VisitorReturnType:
        if out_indices is None:
            out_indices = tuple(frozenset() for _ in self.forward_registry[node])
        ret = super().visit(node, out_indices)
        self.registry[node] = ret
        return ret

    def visit_InnerBroadcast(
        self, node: LAIR.InnerBroadcast, out_indices: Memo
    ) -> VisitorReturnType:

        left_ret = [None for it_out in node.pattern[2]]
        for it_in, dim_in in enumerate(node.pattern[0]):
            for it_out, dim_out in enumerate(node.pattern[2]):
                if dim_in == dim_out:
                    left_ret[it_in] = out_indices[it_out]
        self.visit(node.left, left_ret)

        right_ret = [None for it_out in node.pattern[2]]
        for it_in, dim_in in enumerate(node.pattern[1]):
            for it_out, dim_out in enumerate(node.pattern[2]):
                if dim_in == dim_out:
                    right_ret[it_in] = out_indices[it_out]
        self.visit(node.right, right_ret)

        return out_indices

    def visit_InnerBroadcastMask(
        self, node: LAIR.InnerBroadcast, out_indices: Memo
    ) -> VisitorReturnType:

        left_ret = [None for it_out in node.pattern[2]]
        for it_in, dim_in in enumerate(node.pattern[0]):
            for it_out, dim_out in enumerate(node.pattern[2]):
                if dim_in == dim_out:
                    left_ret[it_in] = out_indices[it_out]
        self.visit(node.left, left_ret)

        right_ret = [None for it_out in node.pattern[2]]
        for it_in, dim_in in enumerate(node.pattern[1]):
            for it_out, dim_out in enumerate(node.pattern[2]):
                if dim_in == dim_out:
                    right_ret[it_in] = out_indices[it_out]
        self.visit(node.right, right_ret)

        return out_indices

    def visit_CastLike(
        self, node: LAIR.CastLike, out_indices: Memo
    ) -> VisitorReturnType:
        self.visit(node.operand, out_indices)
        return out_indices

    def visit_DotMxM(self, node: LAIR.DotMxM, out_indices: Memo) -> VisitorReturnType:
        right_forward = self.forward_registry[node.right]
        left_forward = self.forward_registry[node.left]

        self.visit(node.left, (out_indices[0], right_forward[0]))
        self.visit(node.right, (left_forward[1], out_indices[1]))
        return out_indices

    def visit_DotMxV(self, node: LAIR.DotMxV, out_indices: Memo) -> VisitorReturnType:
        right_forward = self.forward_registry[node.right]
        left_forward = self.forward_registry[node.left]

        self.visit(
            node.left,
            (out_indices[0], right_forward[0]),
        )
        self.visit(
            node.right, (left_forward[1],)
        )
        return out_indices

    def visit_DotVxM(self, node: LAIR.DotVxM, out_indices: Memo) -> VisitorReturnType:
        right_forward = self.forward_registry[node.right]
        left_forward = self.forward_registry[node.left]
        self.visit(
            node.left, (right_forward[0],)
        )
        self.visit(
            node.right, (left_forward[0], out_indices[0]),
        )
        return out_indices

    def visit_ElementwiseAdd(
        self, node: LAIR.ElementwiseAdd, out_indices: Memo
    ) -> VisitorReturnType:
        right_forward = self.forward_registry[node.right]
        left_forward = self.forward_registry[node.left]
        self.visit(
            node.left,
            tuple(
                union_sop(out_indices[it], right_forward[it])
                for it in range(len(out_indices))
            ),
        )
        self.visit(
            node.right,
            tuple(
                union_sop(out_indices[it], left_forward[it])
                for it in range(len(out_indices))
            ),
        )
        return out_indices

    def visit_ElementwiseApply(
        self, node: LAIR.ElementwiseApply, out_indices: Memo
    ) -> VisitorReturnType:
        self.visit(node.operand, out_indices)
        return out_indices

    def visit_Cast(
        self, node: LAIR.Cast, out_indices: Memo
    ) -> VisitorReturnType:
        self.visit(node.operand, out_indices)
        return out_indices

    def visit_ElementwiseApplyBindFirst(
        self, node: LAIR.ElementwiseApplyBindFirst, out_indices: Memo
    ) -> VisitorReturnType:
        self.visit(node.operand, out_indices)
        return out_indices

    def visit_ElementwiseApplyBindSecond(
        self, node: LAIR.ElementwiseApplyBindSecond, out_indices: Memo
    ) -> VisitorReturnType:
        self.visit(node.operand, out_indices)
        return out_indices

    def visit_ElementwiseMultiply(
        self, node: LAIR.ElementwiseMultiply, out_indices: Memo
    ) -> VisitorReturnType:
        right_forward = self.forward_registry[node.right]
        left_forward = self.forward_registry[node.left]
        #print(f"{left_forward=}, {right_forward=}, {out_indices=}")
        if len(left_forward) == len(right_forward):
            self.visit(
                node.left,
                tuple(
                    intersect_sop(out_indices[it], right_forward[it])
                    for it in range(len(out_indices))
                ),
            )
            self.visit(
                node.right,
                tuple(
                    intersect_sop(out_indices[it], left_forward[it])
                    for it in range(len(out_indices))
                ),
            )
        elif len(left_forward) == 0:
            self.visit(node.right, out_indices)
        elif len(right_forward) == 0:
            self.visit(node.left, out_indices)
        return out_indices

    def visit_AntiMask(self, node: LAIR.AntiMask, out_indices: Memo) -> VisitorReturnType:
        operand_forward = self.forward_registry[node.operand]
        mask_forward = self.forward_registry[node.mask]
        self.visit(
            node.mask,
            tuple(
                intersect_sop(out_indices[it], complement_sop(operand_forward[it]))
                for it in range(len(out_indices))
            ),
        )
        self.visit(
            node.operand,
            tuple(
                intersect_sop(out_indices[it], complement_sop(mask_forward[it]))
                for it in range(len(out_indices))
            ),
        )
        return out_indices

    def visit_Mask(self, node: LAIR.Mask, out_indices: Memo) -> VisitorReturnType:
        operand_forward = self.forward_registry[node.operand]
        mask_forward = self.forward_registry[node.mask]
        self.visit(
            node.mask,
            tuple(
                intersect_sop(out_indices[it], operand_forward[it])
                for it in range(len(out_indices))
            ),
        )
        self.visit(
            node.operand,
            tuple(
                intersect_sop(out_indices[it], mask_forward[it])
                for it in range(len(out_indices))
            ),
        )
        return out_indices

    def visit_Output(self, node: LAIR.Output, out_indices: Memo) -> VisitorReturnType:
        self.visit(node.operand, out_indices)
        return out_indices

    def visit_Pattern(
        self, node: LAIR.Pattern, out_indices: Memo
    ) -> VisitorReturnType:
        self.visit(node.operand, out_indices)
        return out_indices

    def visit_PrimaryKey(
        self, node: LAIR.PrimaryKey, out_indices: Memo
    ) -> VisitorReturnType:
        self.visit(node.stencil, out_indices)
        return out_indices

    def visit_Promote(
        self, node: LAIR.Promote, out_indices: Memo
    ) -> VisitorReturnType:
        out_indices_lookup = {
            node.out_indices[it]: indices for it, indices in enumerate(out_indices)
        }
        self.visit(
            node.operand, tuple(out_indices_lookup[idx] for idx in node.operand_indices)
        )
        return out_indices

    def visit_Reduction(
        self, node: LAIR.Reduction, out_indices: Memo
    ) -> VisitorReturnType:
        operand_indices = []
        count = 0
        for it in range(len(out_indices) + len(node.axes)):
            if it in node.axes:
                operand_indices.append(frozenset())
            else:
                operand_indices.append(out_indices[count])
                count += 1
        self.visit(node.operand, tuple(operand_indices))
        return out_indices

    def visit_Tensor(self, node: LAIR.Tensor, out_indices: Memo) -> VisitorReturnType:
        if node.stencil is not None:
            self.visit(node.stencil, out_indices)
        return out_indices

    def visit_PermuteIndices(
        self, node: LAIR.PermuteIndices, out_indices: Memo
    ) -> VisitorReturnType:
        out_indices_lookup = {
            idx: indices for idx, indices in zip(node.new_axes, out_indices)
        }
        self.visit(
            node.operand,
            tuple(out_indices_lookup[idx] for idx in range(len(node.new_axes))),
        )
        return out_indices

    def visit_VectorToColumnMatrix(
        self, node: LAIR.VectorToColumnMatrix, out_indices: Memo
    ):
        self.visit(node.operand, (out_indices[0],))
        return out_indices

    def visit_VectorToRowMatrix(self, node: LAIR.VectorToRowMatrix, out_indices: Memo):
        self.visit(node.operand, (out_indices[1],))
        return out_indices

    def visit_VectorToDiagMatrix(
        self, node: LAIR.VectorToDiagMatrix, out_indices: Memo
    ):
        self.visit(node.operand, (union_sop(out_indices[0], out_indices[1]),))
        return out_indices
