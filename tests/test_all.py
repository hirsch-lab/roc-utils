import sys
import unittest
import numpy as np
from pathlib import Path

import roc_utils._roc as roc

from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score


class TestsIris(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_iris(return_X_y=True,
                                   as_frame=True)
        # Just pick two classes.
        mask = self.y.isin([0,1])
        self.X = self.X[mask]
        self.y = self.y[mask]
        self.pos_label = 1

    def test_sklearn_consistentcy(self):
        X = self.X.iloc[:,0]
        y = self.y
        ret = roc.compute_roc(X=X, y=y, pos_label=self.pos_label)
        auc_sklearn = roc_auc_score(y_true=self.y, y_score=X)
        np.testing.assert_almost_equal(ret.auc, auc_sklearn)


if __name__ == "__main__":
    unittest.main(verbosity=2)
