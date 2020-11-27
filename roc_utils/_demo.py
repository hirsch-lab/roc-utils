import numpy as np
from ._plot import plot_roc_bootstrap, plot_roc
from ._roc import compute_roc

def _sample_data(n1, mu1, std1, n2, mu2, std2, seed=42):
    """
    Construct binary classification problem with n1 and n2
    samples per class, respectively.

    Returns two np.ndarrays x and y of length (n1+n2).
    x represents the predictor, y the binary response.
    """
    rng = np.random.RandomState(seed)
    x1 = rng.normal(mu1, std1, n1)
    x2 = rng.normal(mu2, std2, n2)
    y1 = np.zeros(n1, dtype=bool)
    y2 = np.ones(n2, dtype=bool)
    x = np.concatenate([x1,x2])
    y = np.concatenate([y1,y2])
    return x, y


def demo_basic(n_samples=600, seed=42):
    """
    Demonstrate basic usage of compute_roc() and  plot_roc().
    """
    import matplotlib.pyplot as plt
    pos_label = True
    x, y = _sample_data(n1=n_samples//2, mu1=0.0, std1=0.5,
                        n2=n_samples//2, mu2=1.0, std2=0.7,
                        seed=seed)
    roc = compute_roc(X=x, y=y, pos_label=pos_label)
    plot_roc(roc, label="Dataset", color="red")
    plt.title("Basic demo")
    plt.show()


def demo_bootstrap(n_samples=600, n_bootstrap=50, seed=42):
    """
    Demonstrate a ROC analysis for a bootstrapped dataset.
    """
    import matplotlib.pyplot as plt
    assert(n_samples>2)
    pos_label = True
    x, y = _sample_data(n1=n_samples//2, mu1=0.0, std1=0.5,
                        n2=n_samples//2, mu2=1.0, std2=0.7,
                        seed=seed)
    plot_roc_bootstrap(X=x, y=y, pos_label=pos_label,
                       n_bootstrap=n_bootstrap,
                       random_state=seed+1,
                       show_boots=False,
                       title="Bootstrap demo")
    plt.show()
