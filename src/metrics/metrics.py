import numpy as np
import pandas as pd

from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score
from sksurv.functions import StepFunction

from utils import (
    _compute_similarity,
    compute_survival_mean, 
    compute_survival_function
)

def _calculate_log_likelihood(w, df, times, event, feature_types, similarity_fn, p, q):
    """
    Calculate the log-likelihood of the data given the weights.
    """
    likelihood = np.zeros(df.shape[0])

    for i in range(df.shape[0]):
        X = df[i]
        if event[i] == 1:
            s = _compute_similarity(w, df, X, feature_types, similarity_fn, p, q)
            survival_function = compute_survival_function(w, df, X, times, event, feature_types, similarity_fn, p, q)

            time_i = times[i]

            for k in range(100):
                if (i - k) <= 0:
                    survival = 1
                    break
                else:
                    if times[i - k] != time_i:
                        time_indices = np.where(times == times[i - k])[0]
                        survival = survival_function[time_indices[0]]
                    break

            num = np.sum(s * event * (times == time_i))
            den = np.sum(s * (times >= time_i))
            likelihood[i] = np.log((num / den) * survival)
        
        else:
            time_i = times[i]
            time_indices = np.where(times == time_i)[0]
            survival = compute_survival_function(w, df, X, times, event, feature_types, similarity_fn, p, q)[time_indices[0]]
            likelihood[i] = np.log(survival)

    return -np.sum(likelihood)

def _calculate_concordance_index(w, df, times, event, feature_types, similarity_fn, p, q):
    """
    Calculate the concordance index of the data given the weights.
    """
    ev = np.empty(df.shape[0])

    for i in range(df.shape[0]):
        X = df[i]
        survival_fn = compute_survival_function(w, df, X, times, event, feature_types, similarity_fn, p, q)
        ev[i] = compute_survival_mean(survival_fn, times)
    
    return -concordance_index(times, ev, event)

def _calculate_brier_score(w, df, times, event, feature_types, similarity_fn, p, q):
    """
    Calculate the brier score of the data given the weights.
    """
    survs = np.empty((df.shape[0], len(times)))

    for i in range(df.shape[0]):
        X = df[i]
        sf = compute_survival_function(w, df, X, times, event, feature_types, similarity_fn, p, q)
        survs[i, :] = sf

    transformed_survs = np.array([StepFunction(times, surv) for surv in survs])
    t = np.arange(times.min(), times.max())
    preds = np.asarray([transformed_surv(t) for transformed_surv in transformed_survs])
    
    y_train = pd.DataFrame({'event': event, 'time': times}).to_records(index=False)

    return integrated_brier_score(y_train, y_train, preds, t)
