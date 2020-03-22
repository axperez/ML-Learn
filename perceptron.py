'''
Program: Discrete Percepton Algorithm for Sentiment Analysis (2-word vocab)
'''
# %%
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

# Some functions to plot our points and draw the lines
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

def perceptron_trick(weights, bias, features, label, learning_rate = 0.01):
    pred = prediction(weights, bias, features)
    # feature value tells us what features influenced the prediction more
    for i in range(len(weights)):
        # (label-pred) will either zero out the operand, make it negative, or keep it positive
        weights[i] += (label-pred)*features[i]*learning_rate
    bias += (label-pred)*learning_rate
    return weights, bias

def perceptron_algorithm(X, Y, learning_rate = 0.01, epochs = 200):
    weights = [1.0 for i in range(len(X.loc[0]))]
    bias = 0.0
    errors = []
    for i in range(epochs):
        draw_line(weights[0], weights[1], bias, color='grey', linewidth=1.0, linestyle='dotted')
        errors.append(total_error(weights, bias, X, Y))
        j = random.randint(0, len(X)-1)
        weights, bias = perceptron_trick(weights, bias, X.loc[j], Y[j])
    plot_points(X, Y)
    draw_line(weights[0], weights[1], bias)
    plt.show()
    plt.scatter(range(epochs), errors)
    return weights, bias

if __name__ == "__main__":
    features = pd.DataFrame([[1,0],[0,2],[1,1],[1,2],[1,3],[2,2],[3,2],[2,3]])
    labels = pd.Series([0,0,0,0,1,1,1,1])
    print(perceptron_algorithm(features, labels, epochs=1000))
 

# %%
