import warnings
import numpy as np
import pandas as pd
from scipy import interp

from ._stats import mean_intervals
from ._types import StructContainer
from ._sampling import resample_data


def get_objective(method="minopt", **kwargs):
    """
    The function returns a callable computing a cost f(fpr(c), tpr(c))
    as a function of a cut-off/threshold c. The cost function is used to
    identify an optimal cut-off c* which maximizes this cost function.

    Explanations:
        fpr:    false positive rate == 1 - specificity
        tpr:    true positive rate (recall, sensitivity)

        The diagonal fpr(c)=tpr(c) corresponds to a (diagnostic) test that
        gives the same proportion of positive results for groups with and
        without the condition being true. A perfect test is characterized
        by fpr(c*)=0 and tpr(c*)=1 at the optimal cut-off c*, where there
        are no false negatives (tpr=1) and no false positives (fpr=0).

        The prevalence is the ratio between the cases for which the condition
        is true and the total population:
            prevalence = (tp+fn)/(tp+tn+fp+fn)

    Available methods:
        cost:       Maximizes the distance from the diagonal fpr==tpr.
                    Principally, it is possible to apply particular costs
                    for the four possible outcomes of a diagnostic tests (tp,
                    tn, fp, fn). With C0 being the fixed costs, the C_tp the
                    cost associated with a true positive and P(tp) the
                    proportion of TP's in the population, and so on:
                        C = (C0 + C_tp*P(tp) + C_tn*P(tn)
                             + C_fp*P(fp) + C_fn*P(fn))
                    It can be shown (Metz 1978) that the slope of the ROC curve
                    at the optimal cutoff value is
                        m = (1-prevalence)/prevalence * (C_fp-C_tn)/(C_fn-C_tp)
                    Zweig and Campbell (1993) showed that the point along the
                    ROC curve where the average cost is minimum corresponds to
                    the cutoff value where fm is maximized:
                        J = fm = tpr - m*fpr
                    For m=1, the cost reduces to Youden's index.
        youden:     Computes Youden's J statistic (also called Youden's index).
                        J = sensitivity + specitivity - 1
                          = tpr - fpr
                    Youden's index summarizes the performance of a diagnostic
                    test that is 1 for a perfect test (fpr=0, tpr=1) and 0 for
                    a perfectly useless test (where fpr=tpr). See also the
                    explanations above.
                    Youden's index can be visualized as the distance from the
                    diagonal in vertical direction.
        concordance: Another objective that summarizes the diagnostic
                    performance of a test.
                        J = sensitivity * specitivity
                        J = tpr * (1-fpr)
                    The objective is 0 (minimal) if either (1-fpr) or tpr
                    are 0, and it is 1 (maximal) if tpr=1 and fpr=0.
                    The concordance can be visualized as the rectangular
                    formed by tpr and (1-fpr).
        minopt:     Computes the distance from the optimal point (0,1).
                        J = sqrt((1-specitivity)^2 + (1-sensitivity)^2)
                        J = sqrt(fpr^2 + (1-tpr)^2)
        minoptsym:  Similar as "minopt", but takes the smaller of the distances
                    from points (0,1) and (1,0). This makes sense for a
                    "inverted" predictor whose ROC curve is mainly under the
                    diagonal.
        lr+, plr:   Positive likelihood ratio (LR+):
                        J = tpr / fpr
        lr-, nlr:   Negative likelihood ratio (LR-):
                        J = fnr / tnr
                        J = (1-tpr) / (1-fpr)
        dor:        Diagnostic odds ratio:
                        J = LR+/LR-
                        J = (tpr / fpr) * ((1-fpr) / (1-tpr))
                        J = tpr*(1-fpr) / (fpr*(1-tpr))
        chi2:       An objective proposed by Miller and Siegmund in 1982.
                    The optimal cut-off c* maximizes the standard chi-square
                    statistic with one degree of freedom.
                        J = (tpr-fpr)^2 /
                                ( (P*tpr+N*fpr) / (P+N) *
                                  (1 - (P*tpr+N*fpr) / (P+N)) *
                                  (1/P+1/N)
                                )
                    where P and N are the number of positive and negative
                    observations respectively.
        acc:        Prediction accuracy:
                        J = (TP+TN)/(P+N)
                          = (P*tpr+N*tnr)/(P+N)
                          = (P*tpr+N*(1-fpr))/(P+N)
        cohen:      Cohen kappa ("agreement") between x and y.
                    The goal is to find a threshold thr that binarizes the
                    predictor x such that it maximizes the agreement between
                    observed rater y and that binarized predictor.
                    This function evaluates rather slowly.
                        contingency = [ [TP, FP], [FN, TN]]
                        contingency = [[tpr*P, fpr*N], [(1-tpr)*P, (1-fpr)*N]]
                        J = cohens_kappa(contingency)

        "cost" with m=1, "youden" and "minopt" likely are equivalent in most of
        the cases.

        More about cost functions:
            NCSS Statistical Software: One ROC Curve and Cutoff Analysis:
            https://www.ncss.com/software/ncss/ncss-documentation/#ROCCurves

            Wikipedia: Sensitivity and specificity
            https://en.wikipedia.org/wiki/Sensitivity_and_specificity

            Youden's J statistic
            https://en.wikipedia.org/wiki/Youden%27s_J_statistic

            Rota, Antolini (2014). Finding the optimal cut-point for Gaussian
            and Gamma distributed biomarkers.
            http://doi.org/10.1016/j.csda.2013.07.015

            Unal (2017). Defining an Optimal Cut-Point Value in ROC
            Analysis: An Alternative Approach.
            http://doi.org/10.1155/2017/3762651

            Miller, Siegmund (1982). Maximally Selected Chi Square Statistics.
            http://doi.org/10.2307/2529881
    """
    method = method.lower()
    if method == "cost":
        # Assignment cost ratio, see reference below.
        # It is possible to pass a value for m. Default choice: 1.
        m = kwargs.get("m", 1.)
        def J(fpr, tpr): return tpr - m * fpr
    elif method == "hesse":
        # This function is roughly redundant to "cost".
        # Distance function from diagonal (Hesse normal form)
        # See also: https://ch.mathworks.com/help/stats/perfcurve.html
        m = 1   # Assignment cost ratio, see reference below.
        def J(fpr, tpr): return np.abs(m * fpr - tpr) / np.sqrt(m * m + 1)
    elif method == "youden":
        def J(fpr, tpr): return tpr - fpr
    elif method == "minopt":
        # Take negative distance for maximization.
        def J(fpr, tpr): return -np.sqrt(fpr**2 + (1 - tpr)**2)
    elif method == "minoptsym":
        # Same as minopt, except that it takes the smaller of
        # the distances from points (0,1) or (1,0).
        def J(fpr, tpr): return -min(np.sqrt(fpr**2 + (1 - tpr)**2),
                                     np.sqrt((1 - fpr)**2 + tpr**2))
    elif method in ("plr", "lr+", "positivelikelihoodratio"):
        def J(fpr, tpr): return tpr / fpr if fpr > 0 else -1
    elif method in ("nlr", "lr-", "negativelikelihoodratio"):
        def J(fpr, tpr): return -(1 - tpr) / (1 - fpr) if fpr < 1 else -1
    elif method == "dor":
        def J(fpr, tpr): return tpr * (1 - fpr) / \
            (fpr * (1 - tpr)) if fpr > 0 and tpr < 1 else -1
    elif method == "concordance":
        def J(fpr, tpr): return tpr * (1 - fpr)
    elif method == "chi2":
        # N: number of negative observations.
        # P: number of positive observations.
        assert("N" in kwargs)
        assert("P" in kwargs)
        N = kwargs["N"]
        P = kwargs["P"]
        assert(N > 0 and P > 0)

        def fun(fpr, tpr):
            if (P * tpr + N * fpr) == 0:
                return -1
            if (1 - (P * tpr + N * fpr) / (P + N)) == 0:
                return -1
            return ((tpr - fpr)**2 /
                    (P * tpr + N * fpr) / (P + N) *
                    (1 - (P * tpr + N * fpr) / (P + N)) *
                    (1 / P + 1 / N)
                    )
        J = fun
    elif method == "acc":
        # N: number of negative observations.
        # P: number of positive observations.
        assert("N" in kwargs)
        assert("P" in kwargs)
        N = kwargs["N"]
        P = kwargs["P"]
        def J(fpr, tpr): return (P * tpr + N * (1 - fpr)) / (P + N)
    elif method == "cohen":
        # N: number of negative observations.
        # P: number of positive observations.
        assert("N" in kwargs)
        assert("P" in kwargs)
        N = kwargs["N"]
        P = kwargs["P"]

        def fun(fpr, tpr):
            import warnings
            from statsmodels.stats.inter_rater import cohens_kappa
            contingency = [[(1 - fpr) * N, (1 - tpr) * P],
                           [fpr * N, tpr * P]]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                                        message="invalid value encountered",
                                        category=RuntimeWarning)
                # Degenerate cases are not nicely handled by statsmodels.
                # https://github.com/statsmodels/statsmodels/issues/5530
                return cohens_kappa(contingency)["kappa"]
        J = fun
    return J


def compute_roc(X, y, pos_label=True, objective="minopt", auto_flip=False):
    """
    Compute receiver-operator characteristics for a 1D dataset.

    The ROC curve compares the false positive rate (FPR) with true positive
    rate (TPR) for different classifier thresholds. It is a parametrized curve,
    with the classification threshold being the parameter.

    One can draw the ROC curve with the output of the ROC analysis.
            roc: x=fpr(thr), y=ypr(thr), where thr is the curve parameter
    Parameter thr varies from min(data) to max(data).

    In the context of machine learning, data often represents classification
    probabilities, but it can also be used for any metric data to discriminate
    between two classes.

    Note: The discrimination operation for binary classification is x>thr.
          In case the operation is rather x<thr, the ROC curve is simply
          mirrored along the diagonal fpr=tpr. The computation of the area
          under curve (AUC) takes this into account and returns the maximum
                auc = max(area(fpr,tpr), area(tpr,fpr))

    Arguments:
        X:          The data, pd.Series, a np.ndarray (1D) or a list
        y:          The true labels of the data
        pos_label:  The value of the positive label
        objective:  Identifier for the cost function (see get_objective()),
                    can also be a list of identifiers
        auto_flip:  Set True if an inverted predictor should be flipped
                    automatically upon detection, which will set the
                    roc.inv flag to True. Better approach: switch the
                    pos_label explicitly.

    Returns:
        roc:        A struct consisting of the following elements:
                    - fpr: false positive rate (the "x-vals")
                    - tpr: true positive rate (the "y-vals")
                    - thr: thresholds
                    - auc: area under curve
                    - opd: optimal point(s)
                    - inv: True if inverted predictor was detected (auto_flip)

                    For every specified objective, the opd dictionary
                    contains a struct with the following fields:
                    - ind: index of optimal threshold
                    - opt: the optimal threshold: thr[ind]
                    - opp: the optimal point: (fpr[ind], tpr[ind])
                    - opa: the optimal pred. accuracy (tp+tn)/(len(X))
                    - opq: cost line through optimal point

    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.values                      # ndarray-like numpy array
    else:
        # Assuming an np.ndarray or anything that naturally can be converted
        # into an np.ndarray
        X = np.array(X)

    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.values
    else:
        y = np.array(y)

    # Ensure objective to be a list of strings.
    objectives = [objective] if isinstance(objective, str) else objective
    # Assert X and y are 1d.
    assert(len(X.shape) == 1 or min(X.shape) == 1)
    assert(len(y.shape) == 1 or min(y.shape) == 1)
    X = X.flatten()
    y = y.flatten()
    # The number of labels must match the dimensions (axis is either 0 or 1).
    assert(len(X) == len(y))

    if pd.isna(X).any():
        # msg = "NaNs found in data. Better remove with X[~np.isnan(X)]."
        # warnings.warn(msg)
        isnan = pd.isna(X)
        X = X[~isnan]
        y = y[~isnan]
    if pd.isna(y).any():
        raise RuntimeError("NaNs found in labels.")

    # Convert y into boolean array.
    y = (y == pos_label)

    # Prepare the cost functions.
    p = np.sum(y)
    n = len(y) - p
    costs = {o: get_objective(o, N=n, P=p) for o in objectives}

    # Sort X, and organize the y with the same ordering.
    if True:
        i_sorted = X.argsort()
    else:
        # In case the discriminator is x<thr instead of x>thr, one could
        # simply reverse the sorted values.
        # Note: AUC compensates for this (it simply takes the maximum)
        i_sorted = X.argsort()[::-1]

    X_sorted = X[i_sorted]
    y_sorted = y[i_sorted]
    # Now the thresholds are simply the values d.
    # Because the data has been sorted and it holds that...
    #   fp(thr) = sum(y(d>=thr)==0)
    #   tp(thr) = sum(y(d>=thr)==1)
    # ...one can calculate fp and tp effectively using cumulative sums.

    fp = n - np.cumsum(~y_sorted)
    tp = p - np.cumsum(y_sorted)
    fpr = fp[::-1] / float(n)
    tpr = tp[::-1] / float(p)
    thr = X_sorted[::-1]
    # Remove duplicated thresholds!
    # This fixes the rare case where X[i]-X[i+1]≈tol. np.unique(thr) will
    # keep both X[i] and X[i+1], even though they are almost equal. If
    # some subsequent rounding / cancellation happens, X[i] and X[i+1] might
    # fall together, which will issue a warning compute_roc_aucopt()...
    # We usually don't need this precision anyways, though the fix is hacky.
    thr = thr.round(8)
    # This fixes the issue described in compute_roc_aucopt.
    thr, i_unique = np.unique(thr, return_index=True)
    thr, i_unique = thr[::-1], i_unique[::-1]
    fpr = fpr[i_unique]
    tpr = tpr[i_unique]
    # Finalize (closure).
    thr = np.r_[np.inf, thr, -np.inf]
    fpr = np.r_[0, fpr, 1]
    tpr = np.r_[0, tpr, 1]

    ret = compute_roc_aucopt(fpr=fpr,
                             tpr=tpr,
                             thr=thr,
                             X=X,
                             y=y,
                             costs=costs,
                             auto_flip=auto_flip)
    return ret


def compute_roc_aucopt(fpr, tpr, thr, costs,
                       X=None, y=None, auto_flip=False):
    """
    Given the false positive rates fpr(thr) and true positive rates tpr(thr)
    evaluated for different thresholds thr, the AUC is computed by simple
    integration.

    Besides AUC, the optimal threshold is computed that maximizes some cost
    criteria. Argument costs is expected to be a dictionary with the cost
    type as key and a functor f(fpr, tpr) as value.

    If X and y are provided (optional), the resulting prediction
    accuracy is also computed for the optimal point.

    The function returns a struct as described in compute_roc().
    """
    # It's possible that the last element [-1] occurs more than once sometimes.
    thr_no_last = np.delete(thr, np.argwhere(thr == thr[-1]))
    if len(np.unique(thr_no_last)) != len(thr_no_last):
        warnings.warn("thr should contain only unique values.")

    # Assignment cost ratio, see reference below and get_objective().
    m = 1

    auc = np.trapz(x=fpr, y=tpr)
    flipped = False
    if auto_flip:
        if auc < 0.5:
            auc, flipped = 1 - auc, True      # Mark as flipped.
            fpr, tpr = tpr, fpr             # Flip the data!

    opd = {}
    for cost_id, cost in costs.items():
        # NOTE: The following evaluation is optimistic if x contains duplicated
        #       values! This is best seen in an example. Let's try to optimize
        #       the prediction accuracy acc=(TP+TN)/(T+P). Let x and y be
        #           x = 3 3 3 4 5 6
        #           y = F T T T T T.
        #       The optimal threshold is 3. It binarizes the data as follows,
        #           b = F F F T T T     = x > 3
        #           y = F T T T T T     => acc = 4/6
        #       However, the brute-force optimization permits to find a split
        #       in-between the duplicated values of x.
        #           o = F T T T T T
        #           y = F T T T T T     => acc = 1.0
        # NOTE: The effect of this optimism is negligible if the ordering of
        #       the outcome y for duplicated values in x is randomized. This
        #       is typically the case for natural data.
        #       If the "parametrization" thr does not contain equal points,
        #       the problem is not apparent.
        costs = list(map(cost, fpr, tpr))
        ind = np.argmax(costs)
        opt = thr[ind]
        if X is not None and y is not None:
            # Prediction accuracy (if X available).
            opa = sum((X > opt) == y) / float(len(X))
            opa = (1 - opa) if flipped else opa
        else:
            opa = None
        # Now that we got the index, flip back to extract the data.
        if flipped:
            fpr, tpr = tpr, fpr
        # Characterize optimal point.
        opo = costs[ind]
        opp = (fpr[ind], tpr[ind])
        q = tpr[ind] - m * fpr[ind]
        opq = ((0, 1), (q, m + q))

        opd[cost_id] = StructContainer(ind=ind,  # index of optimal point
                                       opt=opt,  # optimal threshold
                                       opp=opp,  # optimal point (fpr*, tpr*)
                                       opa=opa,  # prediction accuracy
                                       opo=opo,  # objective value
                                       opq=opq)  # line through opt point

    struct = StructContainer(fpr=fpr,
                             tpr=tpr,
                             thr=thr,
                             auc=auc,
                             opd=opd,
                             inv=flipped)
    return struct


def compute_mean_roc(rocs,
                     resolution=101,
                     auto_flip=True,
                     objective=["minopt"]):
    objectives = [objective] if isinstance(objective, str) else objective

    # Initialize
    fpr_mean = np.linspace(0, 1, resolution)
    fpr_mean = np.insert(fpr_mean, 0, 0)  # Insert a leading 0 (closure)
    n_samples = len(rocs)
    thr_all = np.zeros([n_samples, resolution + 1])
    tpr_all = np.zeros([n_samples, resolution + 1])
    auc_all = np.zeros(n_samples)

    # Interpolate the curves to measure at fixed fpr.
    # This is required to compute the mean ROC curve.
    # TODO: Although I'm optimistic, I'm not entirely
    #       sure if it is okay to also interpolate thr.
    for i, ret in enumerate(rocs):
        tpr_all[i, :] = interp(fpr_mean, ret.fpr, ret.tpr)
        thr_all[i, :] = interp(fpr_mean, ret.fpr, ret.thr)
        auc_all[i] = ret.auc
        # Closure
        tpr_all[i, [0, -1]] = ret.tpr[[0, -1]]
        thr_all[i, [0, -1]] = ret.thr[[0, -1]]

    thr_mean = np.mean(thr_all, axis=0)
    tpr_mean = np.mean(tpr_all, axis=0)

    # Compute performance/discriminability measures.
    costs = {o: get_objective(o) for o in objectives}
    ret_mean = compute_roc_aucopt(fpr=fpr_mean,
                                  tpr=tpr_mean,
                                  thr=thr_mean,
                                  X=None,
                                  y=None,
                                  costs=costs,
                                  auto_flip=auto_flip)

    # Invert predictor only if enough evidence was found (-> mean).
    if ret_mean.inv:
        auc_all = 1 - auc_all

    if n_samples > 1:
        auc_mean, auc95_ci, auc95_ti = mean_intervals(auc_all, 0.95)
        auc_std = np.std(auc_all)
    else:
        auc_mean = auc_all[0]
        auc95_ci = auc_mean.copy()
        auc95_ti = auc_mean.copy()
        auc_std = np.zeros_like(auc_mean)
    ret_mean["auc_mean"] = auc_mean
    ret_mean["auc95_ci"] = auc95_ci
    ret_mean["auc95_ti"] = auc95_ti
    ret_mean["auc_std"] = auc_std
    ret_mean["tpr_all"] = tpr_all
    return ret_mean


def compute_roc_bootstrap(X, y, pos_label,
                          objective="minopt",
                          auto_flip=False,
                          n_bootstrap=100,
                          random_state=None,
                          stratified=False,
                          return_mean=True):
    """
    Estimate an average ROC using bootstrap sampling.

    Arguments:
        X:              The data, pd.Series, a np.ndarray (1D) or a list
        y:              The true labels of the data
        pos_label:      See compute_roc()
        objective:      See compute_roc()
        auto_flip:      See compute_roc()
        n_bootstrap:    Number of bootstrap samples to generate.
        random_state:   None, integer or np.random.RandomState
        stratified:     Perform stratified sampling, which takes into account
                        the relative frequency of the labels. This ensures that
                        the samples always will have the same number of
                        positive and negative samples. Enable stratification
                        if the dataset is very imbalanced or small, such that
                        degenerate samples (with only positives or negatives)
                        will become more likely. Disabling this flag results
                        in a mean ROC curve that will appear smoother: the
                        single ROC curves per bootstrap sample show more
                        variation if the total number of positive and negative
                        samples varies, reducing the "jaggedness" of the
                        average curve. Default: False.
        return_mean:    Return only the aggregate ROC-curve instead of a list
                        of n_bootstrap ROC items.

    Returns:
        A list of roc objects (see compute_roc()) or
        a roc object if return_mean=True.
    """
    # n: number of samples
    # k: number of classes
    X = np.asarray(X)
    y = np.asarray(y)
    # n = len(X)
    k = len(np.unique(y))
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    results = []

    # Collect the data. About bootstrapping:
    # https://datascience.stackexchange.com/questions/14369/
    for i in range(n_bootstrap):
        x_boot, y_boot = resample_data(X, y,
                                       replace=True,
                                       stratify=y if stratified else None,
                                       random_state=random_state)

        if len(np.unique(y_boot)) < k:
            # Test for a (hopefully) rare enough situation.
            msg = ("Not all classes are represented in current bootstrap "
                   "sample. Skipping it. If this problem occurs too often, "
                   "try stratified=True or operate with larger samples.")
            warnings.warn(msg)
            continue

        ret = compute_roc(X=x_boot,
                          y=y_boot,
                          pos_label=pos_label,
                          objective=objective,
                          auto_flip=False)
        results.append(ret)

    if return_mean:
        mean_roc = compute_mean_roc(rocs=results,
                                    auto_flip=auto_flip,
                                    objective=objective)
        return mean_roc
    else:
        return results
