from .common import CommonDatabaseUnitTest
from ..test_util import register
from ...util import export
from ... import QIR
import numpy as np

__all__ = []

np.set_printoptions(suppress=True)

@export
@register()
class TestQuery_x_ewisemult_A(CommonDatabaseUnitTest):
    def runTest(self):
        qdb = QIR.QirDatabase(self.db)
        A = qdb.Matrix.aliased("A")
        x = qdb.LeftVector.aliased("x")

        result = (
            qdb.query(A)
                .join(x, A.ridx == x.idx)
                .select(A.ridx, A.cidx, (A.value * x.value).aliased("Result"))
                .group_by(A.ridx, A.cidx)
                .run()
        )


        expected = np.expand_dims(self.np_left_vector, 1) * self.np_matrix
        observed = np.zeros_like(expected)
        result = result.get_column('Result').data
        observed[result.npI, result.npJ] = result.npV

        np.testing.assert_allclose(expected, observed, rtol=1e-4)

@export
@register()
class TestQuery_y_ewisemult_A(CommonDatabaseUnitTest):
    def runTest(self):
        qdb = QIR.QirDatabase(self.db)
        A = qdb.Matrix.aliased("A")
        y = qdb.RightVector.aliased("y")

        result = (
            qdb.query(A)
                .join(y, A.cidx == y.idx)
                .select(A.ridx, A.cidx, (A.value * y.value).aliased("Result"))
                .group_by(A.ridx, A.cidx)
                .run()
        )


        expected = np.expand_dims(self.np_right_vector, 0) * self.np_matrix
        observed = np.zeros_like(expected)
        result = result.get_column('Result').data
        observed[result.npI, result.npJ] = result.npV

        np.testing.assert_allclose(expected, observed, rtol=1e-4)

@export
@register()
class TestQuery_xy_ewisemult_A(CommonDatabaseUnitTest):
    def runTest(self):
        qdb = QIR.QirDatabase(self.db)
        A = qdb.Matrix.aliased("A")
        x = qdb.LeftVector.aliased("x")
        y = qdb.RightVector.aliased("y")

        result = (
            qdb.query(A)
                .join(x, A.ridx == x.idx)
                .join(y, A.cidx == y.idx)
                .select(A.ridx, A.cidx, (A.value * x.value * y.value).aliased("Result"))
                .group_by(A.ridx, A.cidx)
                .run()
        )


        expected = np.expand_dims(self.np_left_vector, 1) * np.expand_dims(self.np_right_vector, 0) * self.np_matrix
        observed = np.zeros_like(expected)
        result_col = result.get_column('Result')
        O = result.get_column('Result').data
        observed[O.npI, O.npJ] = O.npV

        self.assertEqual(result_col.data.nvals, result.stencil.nvals)
        np.testing.assert_allclose(expected, observed, rtol=1e-4)

@export
@register()
class TestQuery_xAy(CommonDatabaseUnitTest):
    def runTest(self):
        qdb = QIR.QirDatabase(self.db)
        A = qdb.Matrix.aliased("A")
        x = qdb.LeftVector.aliased("x")
        y = qdb.RightVector.aliased("y")

        result = (
            qdb.query(A)
                .join(x, A.ridx == x.idx)
                .join(y, A.cidx == y.idx)
                .select(QIR.Sum(A.value * x.value * y.value).aliased("Result"))
                .group_by()
                .run()
        )

        result_col = result.get_column('Result')
        observed = result_col.data[None]
        expected = self.np_left_vector.dot(self.np_matrix).dot(self.np_right_vector)

        self.assertEqual(result.stencil.nvals, 1)
        self.assertEqual(result_col.data.nvals, 1)
        self.assertAlmostEqual(expected, observed, places=4)
