import numpy as np
from .similarity import _compute_similarity

def _compute_area_under_curve(points):
    """
    Calculates the area under the curve using the rectangle method.
    """
    heights = points[:-1, 0]  # Survival values
    bases = np.diff(points[:, 1])  # Difference between consecutive times
    return np.sum(bases * heights)


def compute_survival_mean(survival_fn, times):
    """
    Calculates the mean survival time based on the survival function.
    """
    survival_fn = np.concatenate(([1.0], survival_fn))
    times = np.concatenate(([0.0], times))
    combined_data = np.column_stack((survival_fn, times))
    return _compute_area_under_curve(combined_data)


def compute_survival_median(survival_fn, times):
    """
    Calculates the median survival time based on the survival function.
    """
    combined_data = np.column_stack((survival_fn, times))
    combined_data = combined_data[np.argsort(combined_data[:, 1])]
    below = combined_data[combined_data[:, 0] <= 0.5]
    above = combined_data[combined_data[:, 0] > 0.5]

    if below.size == 0 or above.size == 0:
        return combined_data[-1, 1]  # Returns the longest available time

    a, b = below[-1], above[0]
    median = b[1] + (0.5 - b[0]) * (a[1] - b[1]) / (a[0] - b[0])
    return median


def compute_survival_function(w, df, X, times, event, feature_types, similarity_fn, p, q):
    """
    Calculates the survival function based on the similarity function.
    """
    num_samples, num_features = df.shape
    survival_fn = np.ones(num_samples, dtype=np.float64)
    similarity = _compute_similarity(w, df, X, feature_types, similarity_fn, p, q)
    unique_times, time_counts = np.unique(times, return_counts=True)
    
    prev_survival_fn = 1.0

    for i in range(len(unique_times)):
        t = unique_times[i]
        
        event_mask = (times == t) & (event == 1)
        not_event_mask = (times == t) & (event == 0)
        num = np.sum(similarity * event * (times == t))
        den = np.sum(similarity * (times >= t))
        
        if event_mask.any():
            survival_fn[event_mask] = (1 - num / den) * prev_survival_fn
            survival_fn[not_event_mask] = survival_fn[times == t][0]
        elif not_event_mask.any():
            survival_fn[not_event_mask] = prev_survival_fn
        
        prev_survival_fn = survival_fn[times == t][0]

    return survival_fn