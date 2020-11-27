import numpy as np
import scipy.stats as st


def mean_intervals(data, confidence=0.95, axis=None):
    """
    Compute the mean, the confidence interval of the mean, and the tolerance
    interval. Note that the confidence interval is often misinterpreted [3].

    References:
    [1] https://en.wikipedia.org/wiki/Confidence_interval
    [2| https://en.wikipedia.org/wiki/Tolerance_interval
    [3] https://en.wikipedia.org/wiki/Confidence_interval#Meaning_and_interpretation
    """
    confidence = confidence / 100.0 if confidence > 1.0 else confidence
    assert(0 < confidence < 1)
    a = 1.0 * np.array(data)
    n = len(a)
    # Both s=std() and se=sem() use unbiased estimators (ddof=1).
    m = np.mean(a, axis=axis)
    s = np.std(a, ddof=1, axis=axis)
    se = st.sem(a, axis=axis)
    t = st.t.ppf((1 + confidence) / 2., n - 1)
    ci = np.c_[m - se * t, m + se * t]
    ti = np.c_[m - s * t, m + s * t]
    assert(ci.shape[1] == 2 and ci.shape[0] ==
           np.size(m, axis=None if axis is None else 0))
    assert(ti.shape[1] == 2 and ti.shape[0] ==
           np.size(m, axis=None if axis is None else 0))
    return m, ci, ti


def mean_confidence_interval(data, confidence=0.95, axis=None):
    """
    Compute the mean and the confidence interval of the mean.
    """
    m, ci, _ = mean_intervals(data, confidence, axis=axis)
    return m, ci


def mean_tolerance_interval(data, confidence=0.95, axis=None):
    """
    Compute the tolerance interval for the data.
    """
    m, _, ti = mean_intervals(data, confidence, axis=axis)
    return m, ti
