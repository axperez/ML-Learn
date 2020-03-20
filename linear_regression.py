'''
Program: Linear Regression to Predict House Price
'''
# %%
import random
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

def square_trick(base_price, price_per_room, num_rooms, price, learning_rate = 0.01):
    # y = mx + b
    predicted_price = price_per_room*num_rooms + base_price
    # change y-intercept by a small amount in (price - predicted_price) direction (shifts either upward or downard)
    base_price += learning_rate*(price - predicted_price)
    # change slope to either cause a clockwise or counterclockwise rotation. (price - predicted_price)*num_rooms 
    #    controls which direction the rotation happens depending on placement of point and prediction
    price_per_room += learning_rate*(price-predicted_price)*num_rooms
    return price_per_room, base_price


def plot_points(features, labels):
    x = np.array(features)
    y = np.array(labels)
    plt.scatter(x, y)
    plt.xlabel("Number of Rooms")
    plt.ylabel("House Prices")
    plt.show()

def draw_line(slope, y_intercept, color='black', linewidth=0.7):   
    x = np.linspace(0, 9, 1000)
    plt.plot(x, y_intercept + slope*x, linestyle='-', color=color, linewidth=linewidth)

def linear_regression(features, labels, learning_rate=0.01, epochs = 1000):
    price_per_room = random.random()
    base_price = random.random()
    for i in range(epochs):
        i = random.randint(0, len(features)-1)
        num_rooms = features[i]
        price = labels[i]
        price_per_room, base_price = square_trick(base_price,
                                                  price_per_room,
                                                  num_rooms,
                                                  price,
                                                  learning_rate=learning_rate)
        #draw_line(price_per_room, base_price)
    return price_per_room, base_price


if __name__ == "__main__":
    features = [1, 2, 3, 5, 6, 8, 9]
    labels = [155, 197, 244, 356, 407, 499, 543]
    m, b = linear_regression(features, labels)
    draw_line(m, b)
    plot_points(features, labels)
    # %%
