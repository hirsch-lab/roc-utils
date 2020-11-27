# roc-utils

<!--https://raw.githubusercontent.com/yngvem/group-lasso/master/README.rst-->

<!--[![Downloads](https://pepy.tech/badge/cyminiball)](https://pepy.tech/project/cyminiball)-->
<!--https://pypistats.org/packages/cyminiball-->
[![image](https://img.shields.io/pypi/v/cyminiball.svg)](https://pypi.org/project/cyminiball/)
[![License](https://img.shields.io/pypi/l/cyminiball)](https://github.com/hirsch-lab/cyminiball/blob/main/LICENSE)
[![Build Status](https://travis-ci.org/hirsch-lab/cyminiball.svg?branch=main)](https://travis-ci.org/hirsch-lab/cyminiball)
[![CodeFactor](https://www.codefactor.io/repository/github/hirsch-lab/cyminiball/badge)](https://www.codefactor.io/repository/github/hirsch-lab/cyminiball)
[![DeepSource](https://deepsource.io/gh/hirsch-lab/cyminiball.svg/?label=active+issues)](https://deepsource.io/gh/hirsch-lab/cyminiball/?ref=repository-badge)
<!--Travis build and test-->
<!--Coveralls.io-->
<!--Read-the-docs not required for such a small project-->


This Python package provides tools to compute and visualize [ROC curves](https://en.wikipedia.org/wiki/Receiver_operating_characteristic). ROC curves can be used to graphically assess the diagnostic ability of a binary classifier. 

The package is provided under the MIT license.

### Installation:

    pip install roc-utils

### Usage:

```python
import numpy as np
import matplotlib.pyplot as plt
from roc_utils import *

def sample_data(n1, mu1, std1, n2, mu2, std2, seed=42):
    rng = np.random.RandomState(seed)
    #  sample size, mean, std
    x1 = rng.normal(mu1, std1, n1)
    x2 = rng.normal(mu2, std2, n2)
    y1 = np.zeros(n1, dtype=bool)
    y2 = np.ones(n2, dtype=bool)
    x = np.concatenate([x1,x2])
    y = np.concatenate([y1,y2])
    return x, y

x, y = sample_data(n1=300, mu1=0.0, std1=0.5,
                   n2=300, mu2=1.0, std2=0.7)
pos_label = True
roc = compute_roc(X=x, y=y, pos_label=pos_label)
plot_roc(roc, label="Sample data", color="red")
plt.show()
```

See [examples/examples.ipynb](https://github.com/hirsch-lab/roc-utils/examples/examples.ipynb) for a more detailed introduction.

### Build

To build the package, use the following

```bash
git clone https://github.com/hirsch-lab/roc-utils.git
cd roc-utils
python setup.py sdist bdist_wheel
python tests/test_all.py
python examples/examples.py
```
