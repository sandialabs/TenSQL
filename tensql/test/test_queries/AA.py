import numpy as np

from .common import CommonDatabaseUnitTest
from ..test_util import register
from ...util import export
from ... import QIR

__all__ = []

@export
@register()
class TestQuery_AA(CommonDatabaseUnitTest):
    def runTest(self):
        qdb = QIR.QirDatabase(self.db)
        A = qdb.Matrix.aliased("A")
        B = qdb.Matrix.aliased("B")

        result = (
            qdb.query(A)
                .join(B, A.cidx == B.ridx)
                .select(A.ridx, B.cidx, QIR.Sum(A.value * B.value).aliased("Result"))
                .group_by(A.ridx, B.cidx)
                .run()
        )

        observed = result.get_column('Result').data[:99, :99].to_numpy()
        expected = self.np_matrix.dot(self.np_matrix)

        np.testing.assert_allclose(expected, observed, rtol=1e-4, atol=1e-4)
