# %%
import numpy as np
import pandas as pd

def score(weights, bias, features):
    return features.dot(weights) + bias

def prediction(weights, bias, features):
    return int(score(weights, bias, features) >= 0)

def error(weights, bias, features, label):
    pred = prediction(weights, bias, features)
    if pred == label:
        return 0
    else:
        return np.abs(score(weights, bias, features))

def total_error(weights, bias, X, y):
    total_error = 0
    for i in range(len(X)):
        total_error += error(weights, bias, X.loc[i], y[i])
    return total_error

if __name__ == "__main__":
    features = pd.DataFrame([[1,0],[0,2],[1,1],[1,2],[1,3],[2,2],[3,2],[2,3]])
    labels = pd.Series([0,0,0,0,1,1,1,1])
 

# %%
