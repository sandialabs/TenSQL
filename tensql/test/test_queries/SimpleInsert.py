import numpy as np

from .common import CommonDatabaseUnitTest
from ..test_util import register
from ...util import export
from ... import QIR
from ... import Types

__all__ = []

@export
@register()
class TestInsert_simple(CommonDatabaseUnitTest):
    def runTest(self):
        qdb = QIR.QirDatabase(self.db)
        A = qdb.Matrix

        result = (
            qdb.insert(A).values([
                dict(ridx=99, cidx=0, value=25),
                dict(ridx=0, cidx=99, value=16)
            ]).run()
        )

        observed = result.get_column('value').data[:99, :99].to_numpy()
        expected = np.copy(self.np_matrix)
        expected[99, 0] = 25
        expected[0, 99] = 16

        np.testing.assert_allclose(expected, observed, rtol=1e-4, atol=1e-4)
