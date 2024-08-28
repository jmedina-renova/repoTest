# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:54:48 2024

@author: jmedina
Demo app para visualizar el proceso del descenso del gradiente.
Este algoritmo se encuentra en el núcleo de todo sistema de AI que utilice redes neuronales.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the more complex loss function
def loss_function(x):
    return x**4 - 3*x**3 + 2

# Compute the derivative numerically using finite differences
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# Implement gradient descent with numerical derivative
def gradient_descent_numerical(starting_point, learning_rate, num_iterations):
    x = starting_point
    path = [x]
    for _ in range(num_iterations):
        grad = numerical_derivative(loss_function, x)
        x = x - learning_rate * grad
        path.append(x)
    return path

# Gradient descent parameters
start = 1.5  # starting point
learning_rate = 0.01
iterations = 100

# Run gradient descent
path = gradient_descent_numerical(start, learning_rate, iterations)

# Prepare data for plotting
x_vals = np.linspace(-1, 3, 400)
y_vals = loss_function(x_vals)

fig, ax = plt.subplots()
ax.plot(x_vals, y_vals, label="Loss Function")
point, = ax.plot([], [], 'ro')

def init():
    point.set_data([], [])
    return point,

def update(i):
    x = path[i]
    y = loss_function(x)
    point.set_data(x, y)
    return point,

ani = animation.FuncAnimation(fig, update, frames=len(path), init_func=init, blit=True, repeat=False)

plt.title("Demo Descenso del Gradiente")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
