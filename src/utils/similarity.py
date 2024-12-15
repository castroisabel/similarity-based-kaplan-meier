import numpy as np

def _compute_similarity(w, df, X, feature_types, function, p = None, q = 1):
    """
    Calculates the similarity between the data and the query point.
    """
    num_samples, num_features = df.shape
    distance = np.zeros(num_samples, dtype=np.float64)

    for j in range(num_features):
        if feature_types[j] == 'numerical':
            if p is None:
                distance += w[j] * (np.array(df[:, j], dtype=np.float64) - X[j])**2
            else:
                distance += w[j] * abs(np.array(df[:, j], dtype=np.float64) - X[j])**p
        elif feature_types[j] == 'ordinal':
            range_squared = (max(df[:, j]) - min(df[:, j]))**2
            distance += w[j] * ((np.array(df[:, j], dtype=np.float64) - X[j])**2 / range_squared)
        elif feature_types[j] == 'nominal':
            distance += w[j] * (df[:, j] != X[j]).astype(float)

    if p is not None:
        distance = distance**(1/p) # Only numeric features # to-do: categorical features
    else:
        distance = distance**q # Both numeric and categorical features

    if function == 'EX':
        similarity = np.exp(-distance)
    elif function == 'FR':
        similarity = 1/(1+distance)
    
    return similarity