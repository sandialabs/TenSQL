from .common import CommonDatabaseUnitTest
from ..test_util import register
from ...util import export
from ... import QIR

import numpy as np

__all__ = []

@export
@register()
class TestQuery_PrimaryKeyFilter(CommonDatabaseUnitTest):
    def runTest(self):
        qdb = QIR.QirDatabase(self.db)
        A = qdb.Matrix.aliased("A")

        result = (
            qdb.query(A)
                .select(A.ridx, A.cidx, A.value)
                .group_by(A.ridx, A.cidx)
                .where(A.ridx >= A.cidx)
                .run()
        )

        observed = result.get_column('value').data[:99, :99].to_numpy()
        expected = np.tril(self.np_matrix)

        np.testing.assert_allclose(expected, observed, rtol=1e-4, atol=1e-4)

@export
@register()
class TestQuery_PrimaryKeyFilterPlusOne(CommonDatabaseUnitTest):
    def runTest(self):
        qdb = QIR.QirDatabase(self.db)
        A = qdb.Matrix.aliased("A")

        result = (
            qdb.query(A)
                .select(A.ridx, A.cidx, A.value)
                .group_by(A.ridx, A.cidx)
                .where(A.ridx >= (A.cidx + 1))
                .run()
        )

        observed = result.get_column('value').data[:99, :99].to_numpy()
        expected = np.tril(self.np_matrix, k=-1)

        np.testing.assert_allclose(expected, observed, rtol=1e-4, atol=1e-4)