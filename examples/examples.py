import numpy as np
import matplotlib.pyplot as plt

import context
from roc_utils import *


def sample_data(n1, mu1, std1, n2, mu2, std2, seed=42):
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


def demo_basic_usage():
    """
    Demonstrate basic usage of roc_utils.
    """

    # Construct binary classification problem
    x1, y1 = sample_data(n1=300, mu1=0.0, std1=0.5,
                         n2=300, mu2=1.0, std2=0.7)
    x2, y2 = sample_data(n1=300, mu1=0.2, std1=0.6,
                         n2=300, mu2=0.8, std2=0.7)

    # Show data
    _, (ax1, ax2) = plt.subplots(2,1)
    ax1.hist(x1[~y1], bins=20, density=True,
             color="red", alpha=0.4, label="Class 1")
    ax1.hist(x1[y1], bins=20, density=True,
             color="blue", alpha=0.4, label="Class 2")
    ax1.legend()
    ax1.set_xlabel("x")
    ax1.set_ylabel("density")
    ax1.set_title("Data")
    ax2.hist(x2[~y2], bins=20, density=True,
             color="red", alpha=0.4, label="Class 1")
    ax2.hist(x2[y2], bins=20, density=True,
             color="blue", alpha=0.4, label="Class 2")
    ax2.legend()
    ax2.set_xlabel("x")
    ax2.set_ylabel("density")

    # Compute ROC.
    pos_label = True
    roc1 = compute_roc(X=x1, y=y1, pos_label=pos_label)
    roc2 = compute_roc(X=x2, y=y2, pos_label=pos_label)

    # Access details of ROC object.
    print("Data 1: AUC=%.3f" % (roc1.auc))
    print("Data 2: AUC=%.3f" % (roc2.auc))

    print()
    print("Available ROC data:")
    print(list(roc1.keys()))

    # Plot ROC curve.
    _, ax3 = plt.subplots()
    plot_roc(roc1, label="Data A", color="red", ax=ax3)
    plot_roc(roc2, label="Data B", color="green", ax=ax3)
    # Place the legend outside.
    ax3.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax3.set_title("ROC curves")


def demo_mean_roc():
    """
    Demonstrate to aggregate multiple ROC curves.
    """
    pos_label = True
    n_datasets = 20
    rocs = []
    for i in range(n_datasets):
        x, y = sample_data(n1=300, mu1=0.0, std1=0.5,
                           n2=300, mu2=1.0, std2=0.7, seed=i)
        roc = compute_roc(X=x, y=y, pos_label=pos_label)
        rocs.append(roc)

    # Compute mean ROC curve for a list of ROC containers.
    roc_mean = compute_mean_roc(rocs)

    # One can plot also that mean ROC curve.
    _, ax1 = plt.subplots()
    plot_roc(roc_mean, label="My Mean ROC", color="blue", ax=ax1)

    # Alternatively, use plot_mean_roc() to also visualize confidence
    # and tolerance intervals, CI and TI, respectively.
    plot_mean_roc(rocs, show_ci=True, show_ti=True, ax=ax1)
    ax1.set_title("Mean ROC with CI and TI")

    # One can also show all ROC curves individually.
    _, ax2 = plt.subplots()
    plot_mean_roc(rocs, show_ci=False, show_ti=False, show_all=True, ax=ax2)
    ax2.set_title("Mean ROC and sample ROCs")



def demo_bootstrap_roc():
    pos_label = True
    n_samples = 50
    _, ax1 = plt.subplots()
    x, y = sample_data(n1=300, mu1=0.0, std1=0.5,
                       n2=300, mu2=1.0, std2=0.7)
    plot_roc_bootstrap(X=x, y=y, pos_label=pos_label,
                       n_bootstrap=n_samples,
                       random_state=42,
                       show_boots=False,
                       title="Bootstrap demo",
                       ax=ax1)

    # This is roughly equivalent to the following:
    # _, ax2 = plt.subplots()
    # rocs = compute_roc_bootstrap(X=x, y=y, pos_label=pos_label,
    #                              n_bootstrap=n_samples,
    #                              random_state=42,
    #                              return_mean=False)
    # plot_mean_roc(rocs, show_ci=False, show_ti=False, show_all=True, ax=ax2)
    # ax2.set_title("Bootstrap demo")


def demo_objectives():
    # Note that multiple objective functions can be computed at the same time.
    pos_label = True
    x, y = sample_data(n1=300, mu1=0.0, std1=0.5,
                       n2=300, mu2=1.0, std2=0.7)

    roc = compute_roc(X=x, y=y, pos_label=pos_label,
                      objective=["minopt", "minoptsym", "youden", "cost",
                                 "concordance", "lr+", "lr-", "dor",
                                 "chi2", "acc", "cohen"])

    print()
    print("Comparison of different objectives:")
    for key, val in roc.opd.items():
        print("%15s thr=% .3f, J=%7.3f" % (key+":", val.opt, val.opo) )

    _, ax1 = plt.subplots()
    plot_roc(roc, show_opt=True, ax=ax1)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax1.set_title("Visualization of different optimal points")


def demo_auto_flip():
    # If for some stupid reason, the feature x predicts "the other" label,
    # one can use the argument auto_flip to adjust for this.
    pos_label = True
    x, y = sample_data(n1=300, mu1=0.0, std1=0.5,
                       n2=300, mu2=1.0, std2=0.7)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4.8))
    roc1 = compute_roc(X=x, y=y, pos_label=pos_label)
    roc2 = compute_roc(X=x, y=y, pos_label=not pos_label)
    plot_roc(roc1, show_opt=True, label="Original", color="green", ax=ax1)
    plot_roc(roc2, show_opt=True, label="Flipped", color="red", ax=ax1)
    ax1.set_title("Flipped label with wrong AUC")
    ax1.legend(loc="center")

    roc1 = compute_roc(X=x, y=y, pos_label=pos_label)
    roc2 = compute_roc(X=x, y=y, pos_label=not pos_label, auto_flip=True)
    plot_roc(roc1, show_opt=True, label="Original", color="green", ax=ax2)
    plot_roc(roc2, show_opt=True, label="Flipped", color="red", ax=ax2)
    ax2.set_title("Fixed, with auto-flip")
    legend = ax2.legend(loc="center")


if __name__ == "__main__":
    demo_basic_usage()
    demo_mean_roc()
    demo_bootstrap_roc()
    demo_objectives()
    demo_auto_flip()
    plt.show()
