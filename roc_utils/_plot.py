import numpy as np
from ._roc import compute_roc, compute_mean_roc, compute_roc_bootstrap


def plot_roc(roc,
             color="red",
             label=None,
             show_opt=False,
             show_details=False,
             format_axes=True,
             ax=None,
             **kwargs):
    """
    Plot the ROC curve given the output of compute_roc.

    Arguments:
        roc:            Output of compute_roc() with the following keys:
                        - fpr: false positive rates fpr(thr)
                        - tpr: true positive rates tpr(thr)
                        - opd: optimal point(s).
                        - inv: true if predictor is inverted (predicts ~y)
        label:          Label used for legend.
        show_opt:       Show optimal point.
        show_details:   Show additional information.
        format_axes:    Apply axes settings, show legend, etc.
        kwargs:         A dictionary with detail settings not exposed
                        explicitly in the function signature. The following
                        options are available:
                        - zorder:
                        - legend_out: Place legend outside (default: False)
                        - legend_label_inv: Use 1-AUC if roc.inv=True (True)
                        Additional kwargs are forwarded to ax.plot().
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mplc

    if ax is None:
        ax = plt.gca()

    # Copy the kwargs (a shallow copy should be sufficient).
    # Set some defaults.
    label = label if label else "Feature"
    zorder = kwargs.pop("zorder", 1)
    legend_out = kwargs.pop("legend_out", False)
    legend_label_inv = kwargs.pop("legend_label_inv", True)
    if legend_label_inv:
        auc_disp, auc_val = "1-AUC" if roc.inv else "AUC", roc.auc
    else:
        auc_disp, auc_val = "AUC", roc.auc
    label = "%s (%s=%.3f)" % (label, auc_disp, auc_val)

    # Plot the ROC curve.
    ax.plot(
        roc.fpr,
        roc.tpr,
        color=color,
        zorder=zorder + 2,
        label=label,
        **kwargs)

    # Plot the no-discrimination line.
    label_diag = "No discrimination" if show_details else None
    ax.plot([0, 1], [0, 1], ":k", label=label_diag,
            zorder=zorder, linewidth=1)

    # Visualize the optimal point.
    if show_opt:
        from itertools import cycle
        markers = cycle(["o", "*", "^", "s", "P", "D"])
        for id, opt in roc.opd.items():
            # Some objectives can be visualized.
            # Plot these optional things first.
            if id == "cost":
                # Line parallel to diagonal (shrunk by m if m≠1).
                ax.plot(opt.opq[0], opt.opq[1], ":k",
                        alpha=0.3,
                        zorder=zorder + 1)
            if id == "youden":
                # Vertical line between optimal point and diagonal.
                ax.plot([opt.opp[0], opt.opp[0]],
                        [opt.opp[0], opt.opp[1]],
                        color=color,
                        zorder=zorder + 1)
            if id == "concordance":
                # Rectangle illustrating the area tpr*(1-fpr)
                from matplotlib import patches
                ll = [opt.opp[0], 0]
                w = 1 - opt.opp[0]
                h = opt.opp[1]
                rect = patches.Rectangle(ll, w, h,
                                         facecolor=color,
                                         alpha=0.2,
                                         zorder=zorder + 1)
                ax.add_patch(rect)

            pa_str = (", PA=%.3f" % opt.opa) if opt.opa else ""
            if show_details:
                legend_entry_opt = "Optimal point (%s, thr=%.3g%s)" % (
                    id, opt.opt, pa_str)
            else:
                legend_entry_opt = "Optimal point (thr=%.3g)" % opt.opt

            face_color = mplc.to_rgba(color, alpha=0.3)
            ax.plot(opt.opp[0], opt.opp[1],
                    linestyle="None",
                    marker=next(markers),
                    markerfacecolor=face_color,
                    markeredgecolor=color,
                    label=legend_entry_opt,
                    zorder=zorder + 3)
    if format_axes:
        margin = 0.02
        loc = "upper left" if (roc.inv or legend_out) else "lower right"
        ax.axis("square")
        ax.set_xlim([0 - margin, 1. + margin])
        ax.set_ylim([0 - margin, 1. + margin])
        ax.set_xlabel("FPR (false positive rate)")
        ax.set_ylabel("TPR (true positive rate)")
        ax.grid(True)
        if legend_out:
            ax.legend(loc=loc, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        else:
            ax.legend(loc=loc)


def plot_mean_roc(rocs, auto_flip=True, show_all=False, ax=None, **kwargs):
    """
    Compute and plot the mean ROC curve for a sequence of ROC containers.

    rocs:       List of ROC containers created by compute_roc().
    auto_flip:  See compute_roc(), applies only to mean ROC curve.
    show_all:   If True, show the single ROC curves.
                If an integer, show the rocs[:show_all] roc curves.
    show_ci:    Show confidence interval
    show_ti:    Show tolerance interval
    kwargs:     Forwarded to plot_roc(), applies only to mean ROC curve.

    Optional kwargs argument show_opt can be either False, True or a string
    specifying the particular objective function to be used to plot the
    optimal point. See get_objective() for details. Default choice is the
    "minopt" objective.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    n_samples = len(rocs)

    # Some default values.
    zorder = kwargs.get("zorder", 1)
    label = kwargs.pop("label", "Mean ROC curve")
    # kwargs for plot_roc()...
    show_details = kwargs.get("show_details", False)
    show_opt = kwargs.pop("show_opt", False)
    show_ti = kwargs.pop("show_ti", True)
    show_ci = kwargs.pop("show_ci", True)
    color = kwargs.pop("color", "red")
    is_opt_str = isinstance(show_opt, (str, list, tuple))
    # Defaults for mean-ROC.
    resolution = kwargs.pop("resolution", 101)
    objective = show_opt if is_opt_str else "minopt"

    # Compute average ROC.
    ret_mean = compute_mean_roc(rocs=rocs,
                                resolution=resolution,
                                auto_flip=auto_flip,
                                objective=objective)

    # Plot ROC curve for single bootstrap samples.
    if show_all:
        def isint(x): return isinstance(x, int) and not isinstance(x, bool)
        n_loops = show_all if isint(show_all) else np.inf
        n_loops = min(n_loops, len(rocs))
        for ret in rocs[:n_loops]:
            ax.plot(ret.fpr, ret.tpr,
                    color="gray",
                    alpha=0.2,
                    zorder=zorder + 2)
    if show_ti:
        # 95% interval
        tpr_sort = np.sort(ret_mean.tpr_all, axis=0)
        tpr_lower = tpr_sort[int(0.025 * n_samples), :]
        tpr_upper = tpr_sort[int(0.975 * n_samples), :]
        label_int = "95% of all samples" if show_details else None
        ax.fill_between(ret_mean.fpr, tpr_lower, tpr_upper,
                        color="gray", alpha=.2,
                        label=label_int,
                        zorder=zorder + 1)
    if show_ci:
        # 95% confidence interval
        tpr_std = np.std(ret_mean.tpr_all, axis=0, ddof=1)
        tpr_lower = ret_mean.tpr - 1.96 * tpr_std / np.sqrt(n_samples)
        tpr_upper = ret_mean.tpr + 1.96 * tpr_std / np.sqrt(n_samples)
        label_ci = "95% CI of mean curve" if show_details else None
        ax.fill_between(ret_mean.fpr, tpr_lower, tpr_upper,
                        color=color, alpha=.3,
                        label=label_ci,
                        zorder=zorder)

    # Last but not least, plot the average ROC curve on top  of everything.
    plot_roc(roc=ret_mean, label=label, show_opt=show_opt,
             color=color, ax=ax, zorder=zorder + 3, **kwargs)
    return ret_mean


def plot_roc_simple(X, y, pos_label, auto_flip=True,
                    title=None, ax=None, **kwargs):
    """
    Compute and plot the receiver-operator characteristic curve for X and y.
    kwargs are forwarded to plot_roc(), see there for details.

    Optional kwargs argument show_opt can be either False, True or a string
    specifying the particular objective function to be used to plot the
    optimal point. See get_objective() for details. Default choice is the
    "minopt" objective.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    show_opt = kwargs.pop("show_opt", False)
    is_opt_str = isinstance(show_opt, (str, list, tuple))
    objective = show_opt if is_opt_str else "minopt"
    ret = compute_roc(X=X, y=y, pos_label=pos_label,
                      objective=objective,
                      auto_flip=auto_flip)
    plot_roc(roc=ret, show_opt=show_opt, ax=ax, **kwargs)
    title = "ROC curve" if title is None else title
    ax.get_figure().suptitle(title)
    return ret


def plot_roc_bootstrap(X, y, pos_label,
                       objective="minopt",
                       auto_flip=True,
                       n_bootstrap=100,
                       random_state=None,
                       stratified=False,
                       show_boots=False,
                       title=None,
                       ax=None,
                       **kwargs):
    """
    Similar as plot_roc_simple(), but estimate an average ROC curve from
    multiple bootstrap samples.

    See compute_roc_bootstrap() for the meaning of the arguments.

    Optional kwargs argument show_opt can be either False, True or a string
    specifying the particular objective function to be used to plot the
    optimal point. See get_objective() for details. Default choice is the
    "minopt" objective.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    # 1) Collect the data.
    rocs = compute_roc_bootstrap(X=X, y=y,
                                 pos_label=pos_label,
                                 auto_flip=auto_flip,
                                 n_bootstrap=n_bootstrap,
                                 random_state=random_state,
                                 stratified=stratified,
                                 return_mean=False)
    # 2) Plot the average ROC curve.
    ret_mean = plot_mean_roc(rocs=rocs, auto_flip=auto_flip,
                             show_all=show_boots, ax=ax, **kwargs)

    title = "ROC curve" if title is None else title
    ax.get_figure().suptitle(title)
    ax.set_title("Bootstrap reps: %d, sample size: %d" %
                 (n_bootstrap, len(y)), fontsize=10)
    return ret_mean
