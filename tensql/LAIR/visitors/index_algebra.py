from typing import Optional, Tuple, Set, FrozenSet, Sequence

import operator
import functools
import itertools

AxisIdent = Tuple[Optional[int], int, bool]
POSExpression = FrozenSet[FrozenSet[AxisIdent]]
SOPExpression = FrozenSet[FrozenSet[AxisIdent]]

def complement_axis(ident: AxisIdent):
    return (ident[0], ident[1], not ident[2])

def intersect_sop(*args) -> SOPExpression:
    ret = set()
    args = [arg for arg in args if len(arg) > 0]
    for idx_tuple in itertools.product(*args):
        ret.add(functools.reduce(operator.ior, idx_tuple, frozenset()))
    return frozenset(ret)

def union_sop(*args: Sequence[SOPExpression]) -> SOPExpression:
    return functools.reduce(operator.ior, args, frozenset())

def complement_sop(sop: SOPExpression) -> SOPExpression:
    pos = frozenset(frozenset(complement_axis(ax) for ax in term) for term in sop)
    posop_terms = frozenset(frozenset((term,)) for term in pos)
    return intersect_sop(*posop_terms)
