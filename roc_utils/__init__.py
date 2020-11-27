__version__ = "0.2.0"
__author__ = "Norman Juchler"

from ._roc import (get_objective,
                   compute_roc,
                   compute_mean_roc,
                   compute_roc_bootstrap)
from ._plot import (plot_roc,
                    plot_mean_roc,
                    plot_roc_simple,
                    plot_roc_bootstrap)
from ._demo import (demo_basic,
                    demo_bootstrap)
