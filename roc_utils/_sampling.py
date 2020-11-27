import numpy as np


def resample_data(*arrays, **kwargs):
    """
    Similar to sklearn's resample function, with a few more extras.

    arrays:             Arrays with consistent first dimension.
    kwargs:
        replace:        Sample with replacement. Default: True
        n_samples:      Number of samples. Default: len(arrays[0])
        frac:           Compute the number of samples as a fraction of the
                        array length: n_samples=frac*len(arrays[0])
                        Overrides the value for n_samples if provided.
        random_state:   Determines the random number generation. Can be None,
                        an int or np.random.RandomState. Default: None
        stratify:       An iterable containing the class labels by which the
                        the arrays should be stratified. Default: None
        axis:           Sampling axis. Note: axis!=0 is slow! Also, stratify
                        is currently not supported if axis!=0. Default: axis=0
        squeeze:        Flatten the output array if only one array is provided.
                        Default: Trues
    """
    def _resample(*arrays, replace, n_samples, stratify, rng, axis=0):
        lens = [x.shape[axis] for x in arrays]
        equal_length = (lens.count(lens[0]) == len(lens))
        if not equal_length:
            msg = "Input arrays don't have equal length: %s"
            raise ValueError(msg % lens)
        if stratify is not None:
            msg = "Stratification is not supported yet."
            raise ValueError(msg)
        if not isinstance(rng, np.random.RandomState):
            rng = np.random.RandomState(rng)

        n = lens[0]
        indices = rng.randint(0, n, n_samples)
        # Sampling along an axis!=0 is not very clever.
        arrays = [x.take(indices, axis=axis) for x in arrays]
        # Flatten the output if only one input array was provided.
        return arrays if len(arrays) > 1 else arrays[0]
    try:
        from sklearn.utils import resample
        has_sklearn = True
    except ModuleNotFoundError:
        has_sklearn = False

    replace = kwargs.pop("replace", True)
    n_samples = kwargs.pop("n_samples", None)
    frac = kwargs.pop("frac", None)
    rng = kwargs.pop("random_state", None)
    stratify = kwargs.pop("stratify", None)
    squeeze = kwargs.pop("squeeze", True)
    axis = kwargs.pop("axis", 0)
    if kwargs:
        msg = "Received unexpected argument(s): %s" % kwargs
        raise ValueError(msg)

    arrays = [np.asarray(x) if not hasattr(x, "shape") else x for x in arrays]
    lens = [x.shape[axis] for x in arrays]
    if frac:
        n_samples = int(np.round(frac * lens[0]))
    if n_samples is None:
        n_samples = lens[0]
    if axis > 0 or not has_sklearn:
        ret = _resample(*arrays, replace=replace, n_samples=n_samples,
                        stratify=stratify, rng=rng, axis=axis)
    else:
        ret = resample(*arrays,
                       replace=replace,
                       n_samples=n_samples,
                       stratify=stratify,
                       random_state=rng,)
    # Undo the squeezing, which is done by resample (and _resample).
    if not squeeze and len(arrays) == 1:
        ret = [ret]
    return ret


def bootstrapped(func, data, n_reps, frac,
                 replace=True, stratify=None,
                 random_state=None, alpha=0.05, **settings):
    """
    Execute the func multiple times with randomly drawn subsets of data and
    compute statistics on the test result. The function supports n-ary functors
    if data is a tuple of iterables.

    Arguments:
        func:           Callable with the following signature:
                            func(d1, d2, ..., dn, alpha)      n: len(*data)
        data:           datasets: (d1, d2, ..., dn), arrays with with
                        consistent first dimension.
        n_reps:         number of repetitions
        frac:           fraction of the data length used per iteration, if
                        None, the number of samples is set to len(data[0])
        replace:        sample with replacement
        stratify:       iterable with class labels, or None/False
        random_state:   None, an int or np.random.RandomState
        alpha:          significance level for the test
        settings:       additional arguments are forwarded to the test function
    """
    results = []
    for i in range(n_reps):
        samplee_data = []
        for d in data:
            ret = resample_data(d,
                                frac=frac,
                                replace=replace,
                                stratify=stratify,
                                random_state=random_state)
            sampled_data.append(ret)
        b, p = func(*sampled_data, alpha=alpha, **settings)
        results.append((b, p))
    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0, ddof=1)  # Unbaised estimator.
    count = int(np.sum(results[:, 0]))
    stats = StructContainer(t_mean=mean[0],
                            t_std=std[0],
                            t_count=count,
                            p_mean=mean[1],
                            p_std=std[1],
                            n=len(results))
    return stats
