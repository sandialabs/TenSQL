from typing import Generator, Tuple, Callable, Any, Tuple, Iterable, Dict, Optional
import abc

from .ElementwiseMultiply import ElementwiseMultiply

from .Node import Node
from .Promote import Promote
from .DotMxV import DotMxV
from .DotVxM import DotVxM
from .DotMxM import DotMxM
from .visitors import IndexVisitor, NodeReplacer, ReverseIndexVisitor, ShapeVisitor
from .visitors.index_algebra import intersect_sop
from .. import LAIR
from .. import util

def replace_node(root: Node, old_node: Node, new_node: Node) -> Node:
    #print("Replacing", old_node.dsl)
    #print("with", new_node.dsl)
    visitor = NodeReplacer(old_node, new_node)
    return visitor.visit(root)

def optimize(root: Node, debug: bool = False):
    keep_optimizing = True

    valid_semirings = {
        ("+", "*"),
        ("LOR", "LAND"),
        ("LOR", "FIRST"),
        ("LOR", "SECOND"),
    }

    while keep_optimizing:
        keep_optimizing = False
        replacement = None

        #print("Re-running analysis")

        index_visitor = IndexVisitor()
        index_visitor.visit(root)
        reverse_index_visitor = ReverseIndexVisitor(index_visitor.registry)
        reverse_index_visitor.visit(root)
        shape_visitor = ShapeVisitor()
        # shape_visitor.visit(root)
        for node, node_indices in index_visitor.registry.items():
            if isinstance(node, (LAIR.PermuteIndices,)):
                if tuple(node.new_axes) == tuple(sorted(node.new_axes)):
                    root = replace_node(root, node, node.operand)
                    keep_optimizing = True
            elif isinstance(node, (LAIR.ElementwiseApply,)) and node.op in ('ONE', 'ZERO'):
                # Move elementwise ones and zeros so they are applied before InnerBroadcastMask operations
                broadcast = node.operand
                if isinstance(broadcast, (LAIR.InnerBroadcastMask,)):
                    operand = broadcast.left
                    replacement = LAIR.InnerBroadcastMask(
                        LAIR.ElementwiseApply(node.op, operand),
                        broadcast.left_indices,
                        broadcast.right,
                        broadcast.right_indices,
                        broadcast.out_indices
                    )
                    root = replace_node(root, node, replacement)
                    keep_optimizing = True
                    replacement = None
            elif isinstance(node, (LAIR.ElementwiseMultiply, LAIR.Mask, LAIR.AntiMask)) and node.op in ("FIRST", "SECOND"):
                #if debug:
                #    print("Optimizing", node.description)
                #    print(node.description, "Forward\n\t", index_visitor.registry[node], "\n\t", index_visitor.registry[node.left], "\n\t", index_visitor.registry[node.right], "\n")
                #    print(node.description, "Reverse\n\t", reverse_index_visitor.registry[node], "\n\t", reverse_index_visitor.registry[node.left], "\n\t", reverse_index_visitor.registry[node.right], "\n")

                if node.op == "FIRST":
                    combined_operand_indices = tuple(
                        intersect_sop(
                            forward_left_idx, reverse_out_idx
                        )
                        for forward_left_idx, reverse_out_idx in zip(
                            index_visitor.registry[node.left],
                            reverse_index_visitor.registry[node],
                        )
                    )

                    redundant = all(
                        all(
                            any(
                                right_index_term <= out_index_term
                                for out_index_term in out_idxset
                            )
                            for right_index_term in right_idxset
                        )
                        for right_idxset, out_idxset in zip(
                            index_visitor.registry[node.right], combined_operand_indices
                        )
                    )
                    unnecessary = node_indices == index_visitor.registry[node.left]
                    if unnecessary or redundant:
                        #print("Removing unnecessary", node.description, redundant, unnecessary)
                        root = replace_node(root, node, node.left)
                        keep_optimizing = True
                elif node.op == "SECOND":
                    combined_operand_indices = tuple(
                        intersect_sop(
                            forward_right_idx, reverse_out_idx
                        )
                        for forward_right_idx, reverse_out_idx in zip(
                            index_visitor.registry[node.right],
                            reverse_index_visitor.registry[node],
                        )
                    )

                    redundant = all(
                        all(
                            any(
                                left_index_term <= out_index_term
                                for out_index_term in out_idxset
                            )
                            for left_index_term in left_idxset
                        )
                        for left_idxset, out_idxset in zip(
                            index_visitor.registry[node.left], combined_operand_indices
                        )
                    )
                    unnecessary = node_indices == index_visitor.registry[node.right]
                    # print(redundant, unnecessary)
                    if unnecessary or redundant:
                        #print("Removing unnecessary SECOND", redundant, unnecessary)
                        root = replace_node(root, node, node.right)
                        keep_optimizing = True
                elif isinstance(node.left, (LAIR.InnerBroadcastMask, LAIR.InnerBroadcast)) and isinstance(node.right, (LAIR.InnerBroadcastMask, LAIR.InnerBroadcast)) and node.left.op == "FIRST" and node.right.op == "FIRST" and node.left.right is node.right.right and node.op not in util.BinaryBooleanGrbOps:
                    left_decoder = dict(zip(node.left.out_indices, range(len(node.left.out_indices))))
                    right_decoder = dict(zip(node.right.out_indices, range(len(node.right.out_indices))))

                    out_indices = tuple(range(len(left_decoder)))
                    left_indices = tuple(left_decoder[idx] for idx in node.left.left_indices)
                    right_indices = tuple(right_decoder[idx] for idx in node.right.left_indices)

                    #print("&"*80)
                    #print(node.op)
                    #print("&"*80)

                    later_mask = LAIR.Mask(
                        LAIR.InnerBroadcast(node.op, node.left.left, left_indices, node.right.left, right_indices, out_indices),
                        node.right.right
                    )
                    root = replace_node(root, node, later_mask)
                    keep_optimizing = True
            elif isinstance(node, LAIR.InnerBroadcastMask):
                if set(node.left_indices) == set(node.out_indices) and set(node.right_indices) == set(node.out_indices):
                    left, right = node.left, node.right
                    if node.left_indices != node.out_indices:
                        new_indices = [list(node.out_indices).index(i) for i in node.left_indices]
                        left = LAIR.PermuteIndices(left, new_indices)
                    if node.right_indices != node.out_indices:
                        new_indices = [list(node.out_indices).index(i) for i in node.right_indices]
                        right = LAIR.PermuteIndices(right, new_indices)
                    root = replace_node(root, node, LAIR.Mask(left, right))
                    keep_optimizing = True
            elif isinstance(node, LAIR.InnerBroadcast):
                #print("Checking", node.pattern, isinstance(node.left, (LAIR.InnerBroadcastMask, LAIR.InnerBroadcast)))
                if set(node.left_indices) == set(node.out_indices) and set(node.right_indices) == set(node.out_indices):
                    left, right = node.left, node.right
                    if node.left_indices != node.out_indices:
                        left = LAIR.PermuteIndices(left, node.left_indices)
                    if node.right_indices != node.out_indices:
                        right = LAIR.PermuteIndices(right, node.right_indices)
                    root = replace_node(root, node, LAIR.ElementwiseMultiply(node.op, left, right))
                    keep_optimizing = True
                elif isinstance(node.left, (LAIR.InnerBroadcastMask, LAIR.InnerBroadcast)) and node.left.op == "FIRST" and len(node.left.right_indices) == len(node.out_indices):
                    left_out_encoder = dict(zip(node.left.out_indices, range(len(node.left.out_indices))))
                    left_out_decoder = dict(zip(range(len(node.left.out_indices)), node.left.out_indices))
                    out_encoder = dict(zip(node.out_indices, range(len(node.out_indices))))
                    out_decoder = dict(zip(range(len(node.out_indices)), node.out_indices))

                    out_indices = tuple(range(len(node.out_indices)))
                    left_indices = tuple(out_encoder[left_out_encoder[idx]] for idx in node.left.left_indices)
                    right_indices = tuple(out_encoder[idx] for idx in node.right_indices)
                    mask_indices = tuple(out_encoder[left_out_encoder[idx]] for idx in node.left.right_indices)

                    later_mask = LAIR.Mask(
                        LAIR.InnerBroadcast(node.op, node.left.left, left_indices, node.right, right_indices, out_indices),
                        LAIR.PermuteIndices(node.left.right, mask_indices)
                    )
                    root = replace_node(root, node, later_mask)
                    keep_optimizing = True
                elif isinstance(node.right, (LAIR.InnerBroadcastMask, LAIR.InnerBroadcast)) and node.right.op == "FIRST" and len(node.right.right_indices) == len(node.out_indices):
                    right_out_encoder = dict(zip(node.right.out_indices, range(len(node.right.out_indices))))
                    right_out_decoder = dict(zip(range(len(node.right.out_indices)), node.right.out_indices))
                    out_encoder = dict(zip(node.out_indices, range(len(node.out_indices))))
                    out_decoder = dict(zip(range(len(node.out_indices)), node.out_indices))

                    out_indices = tuple(range(len(node.out_indices)))
                    right_indices = tuple(out_encoder[right_out_encoder[idx]] for idx in node.right.left_indices)
                    left_indices = tuple(out_encoder[idx] for idx in node.left_indices)
                    mask_indices = tuple(out_encoder[right_out_encoder[idx]] for idx in node.right.right_indices)

                    later_mask = LAIR.Mask(
                        LAIR.InnerBroadcast(node.op, node.left, left_indices, node.right.left, right_indices, out_indices),
                        LAIR.PermuteIndices(node.right.right, mask_indices)
                    )
                    root = replace_node(root, node, later_mask)
                    keep_optimizing = True
            elif isinstance(node, LAIR.ElementwiseMultiply):
                if isinstance(node.left, (LAIR.InnerBroadcastMask, LAIR.InnerBroadcast)) and node.left.op == "FIRST" and node.op not in util.BinaryBooleanGrbOps:
                    left_decoder = dict(zip(node.left.out_indices, range(len(node.left.out_indices))))
                    left_indices = tuple(left_decoder[idx] for idx in node.left.left_indices)
                    out_indices = tuple(range(len(left_decoder)))
                    right_indices = out_indices

                    later_mask = LAIR.Mask(
                        LAIR.InnerBroadcast(node.op, node.left.left, left_indices, node.right, right_indices, out_indices),
                        node.left.right
                    )
                    root = replace_node(root, node, later_mask)
                    keep_optimizing = True
                elif isinstance(node.right, (LAIR.InnerBroadcastMask, LAIR.InnerBroadcast)) and node.right.op == "FIRST" and node.op not in util.BinaryBooleanGrbOps:
                    right_decoder = dict(zip(node.right.out_indices, range(len(node.right.out_indices))))
                    right_indices = tuple(right_decoder[idx] for idx in node.right.left_indices)
                    out_indices = tuple(range(len(right_decoder)))
                    left_indices = out_indices

                    later_mask = LAIR.Mask(
                        LAIR.InnerBroadcast(node.op, node.left, left_indices, node.right.left, right_indices, out_indices),
                        node.right.right
                    )
                    root = replace_node(root, node, later_mask)
                    keep_optimizing = True
            elif isinstance(node, LAIR.Reduction):
                reduction = node
                broadcast = None
                operand = None

                if isinstance(node.operand, (LAIR.InnerBroadcast, LAIR.InnerBroadcastMask)):
                    broadcast = node.operand
                    operand = broadcast.left
#                elif isinstance(node.operand, (LAIR.ElementwiseApply,)):
#                    if isinstance(node.operand.operand, (LAIR.InnerBroadcast, LAIR.InnerBroadcastMask)) and node.operand.operand.op == 'FIRST':
#                        ewiseapply = node.operand
#                        broadcast = ewiseapply.operand
#                        operand = LAIR.ElementwiseApply(ewiseapply.op, broadcast.left)


                if isinstance(broadcast, (LAIR.InnerBroadcast, LAIR.InnerBroadcastMask)):
                    add_op, mul_op = node.op, broadcast.op
                    replacement = None
                    if debug:
                        print("")
                        print("=" * 120)
                        print(add_op, mul_op, broadcast.pattern, (add_op, mul_op) in valid_semirings, broadcast.pattern == ((0, 2), (2, 1), (0, 1, 2)))
                        print(reduction.dsl)
                        print("=" * 120)
                    if (add_op, mul_op) in valid_semirings:
                        remaining_reduction_axes = set(reduction.axes)

                        def is_matrix_multiply(pattern, reduction_axes):
                            if tuple(len(x) for x in pattern) != (2, 2, 3):
                                return False
                            elif set(pattern[0]) == set(pattern[1]):
                                return False

                            shared_items = set(pattern[0]) & set(pattern[1])
                            if len(shared_items) != 1:
                                return False

                            shared_item = tuple(shared_items)[0]
                            if shared_item not in reduction_axes:
                                return False
                            
                            return True

                        #print(f"{is_matrix_multiply(broadcast.pattern, reduction.axes)=}")
                        if is_matrix_multiply(broadcast.pattern, reduction.axes):
                            shared_item = tuple(set(broadcast.pattern[0]) & set(broadcast.pattern[1]))[0]
                            
                            left, right = operand, broadcast.right
                            if shared_item == broadcast.pattern[0][0]:
                                left = LAIR.PermuteIndices(left, (1, 0))
                            if shared_item == broadcast.pattern[1][1]:
                                right = LAIR.PermuteIndices(right, (1, 0))

                            remaining_reduction_axes -= {shared_item}
                            replacement = LAIR.DotMxM(left, right, add_op, mul_op)
#                        elif broadcast.pattern == ((0, 2), (2, 1), (0, 1, 2)) and 2 in reduction.axes:
#                            replacement = LAIR.DotMxM(
#                                operand, broadcast.right, add_op, mul_op
#                            )
#                            remaining_reduction_axes -= {2,}
#                        elif broadcast.pattern == ((0, 2), (1, 2), (0, 1, 2)) and 2 in reduction.axes:
#                            replacement = LAIR.DotMxM(
#                                operand, LAIR.PermuteIndices(broadcast.right, (1, 0)), add_op, mul_op
#                            )
#                            remaining_reduction_axes -= {2,}
#                        elif broadcast.pattern == ((2, 0), (1, 2), (0, 1, 2)) and 2 in reduction.axes:
#                            replacement = LAIR.DotMxM(
#                                LAIR.PermuteIndices(operand, (1, 0)),
#                                LAIR.PermuteIndices(broadcast.right, (1, 0)),
#                                add_op, mul_op
#                            )
#                            remaining_reduction_axes -= {2,}
#                        elif broadcast.pattern == ((2, 0), (2, 1), (0, 1, 2)) and 2 in reduction.axes:
#                            replacement = LAIR.DotMxM(
#                                LAIR.PermuteIndices(operand, (1, 0)),
#                                broadcast.right,
#                                add_op, mul_op
#                            )
#                            remaining_reduction_axes -= {2,}
                        elif (len(broadcast.pattern[0]), len(broadcast.pattern[1]), len(broadcast.pattern[2])) in ((2, 1, 2), (1, 2, 2)) and (0, 1) == reduction.axes:
                            if len(broadcast.pattern[0]) == 2:
                                matrix, vector = operand, broadcast.right
                                matrix_pattern, vector_pattern, out_pattern = broadcast.pattern
                            else:
                                vector, matrix = operand, broadcast.right
                                vector_pattern, matrix_pattern, out_pattern = broadcast.pattern

                            n_matrix = LAIR.Reduction(
                                reduction.op, matrix,
                                axes=tuple(set(matrix_pattern) - set(vector_pattern))
                            )
                            replacement = LAIR.InnerBroadcast(
                                mul_op, n_matrix, (0,), vector, (0,), (0,)
                            )
                            remaining_reduction_axes = {0}
                        elif (len(broadcast.pattern[0]), len(broadcast.pattern[1]), len(broadcast.pattern[2])) in ((2, 1, 2),) and broadcast.pattern[1] == reduction.axes:
                            if broadcast.pattern[0] == (0, 1):
                                matrix = operand
                            else:
                                matrix = LAIR.PermuteIndices(operand)
                            vector = broadcast.right

                            if broadcast.pattern[1] == (0,):
                                replacement = LAIR.DotVxM(
                                    vector, matrix, add_op, mul_op
                                )
                            elif broadcast.pattern[1] == (1,):
                                replacement = LAIR.DotMxV(
                                    matrix, vector, add_op, mul_op
                                )
                            else:
                                assert False

                            remaining_reduction_axes = set()
                        elif (len(broadcast.pattern[0]), len(broadcast.pattern[1]), len(broadcast.pattern[2])) in ((1, 2, 2),) and broadcast.pattern[0] == reduction.axes:
                            if broadcast.pattern[1] == (0, 1):
                                matrix = broadcast.right
                            else:
                                matrix = LAIR.PermuteIndices(broadcast.right)
                            vector = operand

                            if broadcast.pattern[0] == (0,):
                                replacement = LAIR.DotVxM(
                                    vector, matrix, add_op, mul_op
                                )
                            elif broadcast.pattern[0] == (1,):
                                replacement = LAIR.DotMxV(
                                    matrix, vector, add_op, mul_op
                                )
                            else:
                                assert False

                            remaining_reduction_axes = set()
                        elif broadcast.pattern == ((0, 1), (1, 2), (0, 1, 2)) and 1 in reduction.axes:
                            replacement = LAIR.DotMxM(
                                operand, broadcast.right, add_op, mul_op
                            )
                            remaining_reduction_axes -= {1,}
                        elif broadcast.pattern == ((1,), (1, 0), (0, 1)) and 1 in reduction.axes:
                            replacement = LAIR.DotVxM(
                                operand, broadcast.right, add_op, mul_op
                            )
                            remaining_reduction_axes -= {1,}
                        elif broadcast.pattern == ((1,), (0, 1), (0, 1)) and 1 in reduction.axes:
                            replacement = LAIR.DotVxM(
                                operand, LAIR.PermuteIndices(broadcast.right), add_op, mul_op
                            )
                            remaining_reduction_axes -= {1,}
                        elif broadcast.pattern == ((0,1), (1,), (0,1)) and 1 in reduction.axes:
                            replacement = LAIR.DotMxV(
                                operand, broadcast.right, add_op, mul_op
                            )
                            remaining_reduction_axes -= {1,}
                        elif broadcast.pattern == ((1,0), (1,), (0,1)) and 1 in reduction.axes:
                            replacement = LAIR.DotMxV(
                                LAIR.PermuteIndices(operand), broadcast.right, add_op, mul_op
                            )
                            remaining_reduction_axes -= {0,}

                        if replacement is not None:
                            #if ewiseapply is not None:
                            #    replacement = LAIR.ElementwiseApply(ewiseapply.op, replacement)

                            new_reduce_axes = set()
                            removed_count = 0
                            for ax in reduction.axes:
                                if ax in remaining_reduction_axes:
                                    new_reduce_axes.add(ax - removed_count)
                                else:
                                    removed_count += 1

                            if len(remaining_reduction_axes) > 0:
                                replacement = LAIR.Reduction(reduction.op, replacement, tuple(new_reduce_axes))

                if replacement is not None:
                    root = replace_node(root, node, replacement)
                    keep_optimizing = True
            if keep_optimizing:
                break

    return root
