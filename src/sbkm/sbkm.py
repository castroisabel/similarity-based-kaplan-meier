import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score
from sksurv.functions import StepFunction

from utils import (
    _prepare_survival_data,
    _compute_similarity,
    compute_survival_mean,
    compute_survival_median,
    compute_survival_function
)

from metrics import (
    _calculate_log_likelihood,
    _calculate_concordance_index,
    _calculate_brier_score
)


class SimilarityBasedKM():
    def fit(self, df, times, event, feature_types, optimization_fn='likelihood', similarity_fn='EX', p=None, q=1, w0=None):
        if w0 is None:
            w0 = np.zeros(df.shape[1])
        return self._fit(df, times, event, feature_types, optimization_fn, similarity_fn, p, q, w0) 

    def _fit(self, df, times, event, feature_types, optimization_fn, similarity_fn, p, q, w0):
        (self.df,
         self.times,
         self.event,
         self.feature_types) = _prepare_survival_data(df, times, event, feature_types)
        
        self.w0 = w0
        self.p = p
        self.q = q
        self.optimization_fn = optimization_fn
        self.similarity_fn = similarity_fn
        self.weights = self._estimate_weight()
        self.concordance_index_ = self.calculate_concordance_index()
        self.brier_score_ = self.calculate_brier_score()
        self.likelihood_ = self.calculate_likelihood()
        
        return self
    

    # Optimization methods
    def _estimate_weight(self):
        m = self.df.shape[1]
        bounds = [(0, None)]*m
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 100},) #normalization constraint

        if self.optimization_fn == 'likelihood':
            function = _calculate_log_likelihood
        elif self.optimization_fn == 'brier_score':
            function = _calculate_brier_score

        res = minimize(
                    function,
                    self.w0, 
                    args=(self.df, self.times, self.event, self.feature_types, self.similarity_fn, self.p, self.q), 
                    bounds=bounds, 
                    constraints=cons)
        
        return res.x


    # Optimization functions
    def calculate_likelihood(self):
        return np.exp(-_calculate_log_likelihood(self.weights, self.df, self.times, self.event, self.feature_types, self.similarity_fn, self.p, self.q))

    def calculate_concordance_index(self):
        return -_calculate_concordance_index(self.weights, self.df, self.times, self.event, self.feature_types, self.similarity_fn, self.p, self.q)

    def calculate_brier_score(self):
        return _calculate_brier_score(self.weights, self.df, self.times, self.event, self.feature_types, self.similarity_fn, self.p, self.q)


    # Prediction methods
    def predict_survival_function(self, X):
        return compute_survival_function(self.weights, self.df, X, self.times, self.event, self.feature_types, self.similarity_fn, self.p, self.q)
    
    def predict_mean_survival_time(self, X):
        return compute_survival_mean(self.predict_survival_function(X), self.times)
    
    def predict_median_survival_time(self, X):
        return compute_survival_median(self.predict_survival_function(X), self.times)


    # Plotting methods
    def plot_survival_function(self, X, title=' ', label='Similarity-Based KM'):
        survival_fn = np.concatenate(([1.0], self.predict_survival_function(X)))
        times = np.concatenate(([0.0], self.times))
        plt.rcParams['axes.titlesize'] = 25
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        plt.rcParams['axes.labelsize'] = 22
        plt.rcParams['legend.fontsize'] = 22
        plt.rcParams['figure.figsize'] = (13, 8)
        plt.step(times, survival_fn, where='post', linestyle='--', color = 'red', label=label, linewidth=2)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\widehat{S}(t)$')
        plt.title(title)
        plt.xlim(-0.5, max(times)+0.5)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.show()


    # Evaluation methods
    def calculate_similarity(self, X):
        return _compute_similarity(self.weights, self.df, X, self.feature_types, self.similarity_fn, self.p, self.q)
    
    def print_summary(self, X_test, T_test, E_test):
        survs = []
        evs = []

        survs = [self.predict_survival_function(X_test[i]) for i in range(X_test.shape[0])]
        evs = [self.predict_mean_survival_time(X_test[i]) for i in range(X_test.shape[0])]

        transformed_survs = np.array([StepFunction(self.times, surv) for surv in survs])
        t = np.arange(max(self.times.min(), T_test.min()), min(self.times.max(), T_test.max()))
        preds = np.asarray([transformed_surv(t) for transformed_surv in transformed_survs])

        y_train = pd.DataFrame({'event': self.event, 'time': self.times}).to_records(index=False)
        y_test = pd.DataFrame({'event': E_test, 'time': T_test}).to_records(index=False)

        return pd.DataFrame({
            'Weight': [self.weights],
            'Train CI': self.concordance_index_,
            'Test CI': concordance_index(T_test, evs, E_test),
            'Train BS': self.brier_score_,
            'Test BS': integrated_brier_score(y_train, y_test, preds, t),
            'Likelihood': self.likelihood_,
            'SD': np.std(evs)
        })
