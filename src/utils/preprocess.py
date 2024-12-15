import numpy as np
import pandas as pd

def _prepare_survival_data(df, duration, event, features_types):
    """
    Prepares data for survival analysis by ordering by time and event.
    """
    y = pd.concat([duration, event], axis=1, keys=[duration.name, event.name])
    y.sort_values(by=[duration.name, event.name], ascending=[True, False], inplace=True)
    df = df.loc[y.index].values
    duration = y[duration.name].values
    event = y[event.name].values
    features_types = np.array(features_types)
    return df, duration, event, features_types
