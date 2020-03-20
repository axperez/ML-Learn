'''
Program: Continuous Percepton Logistic Regression for Sentiment Analysis (2-word vocab)
'''
# %%
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

def plot_points(features, labels):
    X = np.array(features)
    y = np.array(labels)
    spam = X[np.argwhere(y==1)]
    ham = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in spam],
                [s[0][1] for s in spam],
                s = 25,
                color = 'cyan',
                edgecolor = 'k',
                marker = '^')
    plt.scatter([s[0][0] for s in ham],
                [s[0][1] for s in ham],
                s = 25,
                color = 'red',
                edgecolor = 'k',
                marker = 's')
    plt.xlabel('aack')
    plt.ylabel('beep')
    plt.legend(['happy','sad'])

def draw_line(a,b,c, color='black', linewidth=2.0, linestyle='solid', starting=0, ending=3):
    # Plotting the line ax + by + c = 0
    x = np.linspace(starting, ending, 1000)
    plt.plot(x, -c/b - a*x/b, linestyle=linestyle, color=color, linewidth=linewidth)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def lr_prediction(weights, bias, features):
    return sigmoid(score(weights, bias, features))

def score(weights, bias, features):
    return features.dot(weights) + bias

def log_loss(weights, bias, features, label):
    prediction = lr_prediction(weights, bias, features)
    return label*np.log(prediction) + (1-label)*np.log(1-prediction)

def total_log_loss(weights, bias, X, y):
    total_error = 0
    for i in range(len(X)):
        total_error -= log_loss(weights, bias, X.loc[i], y[i])
    return total_error

def lr_trick(weights, bias, features, label, learning_rate = 0.01):
    pred = lr_prediction(weights, bias, features)
    for i in range(len(weights)):
        weights[i] += (label-pred)*features[i]*learning_rate
        bias += (label-pred)*learning_rate
    return weights, bias
 
def lr_algorithm(features, labels, learning_rate = 0.01, epochs = 200):
    weights = [1.0 for i in range(len(features.loc[0]))]
    bias = 0.0
    errors = []
    for i in range(epochs):
        #draw_line(weights[0], weights[1], bias, color='grey', linewidth=1.0, linestyle='dotted')
        errors.append(total_log_loss(weights, bias, features, labels))
        j = random.randint(0, len(features)-1)
        weights, bias = lr_trick(weights, bias, features.loc[j], labels[j])
    draw_line(weights[0], weights[1], bias)
    plot_points(features, labels)
    plt.show()
    plt.scatter(range(epochs), errors)
    plt.xlabel('epochs')
    plt.ylabel('error')
    return weights, bias

if __name__ == "__main__": 
    X = pd.DataFrame([[1,0],[0,2],[1,1],[1,2],[1,3],[2,2],[3,2],[2,3]])
    y = pd.Series([0,0,0,0,1,1,1,1])
    lr_algorithm(X, y, epochs=3000)

# %%
