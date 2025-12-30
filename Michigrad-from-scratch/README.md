# Michigrad
A small, educational Autograd engine created by Dr. Joaquin Bogado for his course "Deep Learning Concepts for Text Generative AI".

![gatite](images/gatite.png)

This is a clone of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) and basically shares the same codebase. Some visualization aspects have been improved and tailored for the *"Deep Learning Concepts for Text Generative AI"* course at [Purrfect AI](https://purrfectai.online).

## Features
Michigrad is a gradient calculation engine for scalar values. It allows you to represent numeric values by wrapping them in `Value` objects. These objects support operations analogous to standard numbers, such as addition, multiplication, division, and exponentiation, among others. 

Michigrad allows you to compute the result of applying these operations to `Values` (known as the **forward pass**), but it also generates a graph of operations and dependencies required to reach the result. This graph can be used to calculate the gradients of any `Value` in the graph with respect to the result using the **backpropagation** algorithm, which Michigrad also implements. 

This information can be used to modify the weights $W$ of a neural network with respect to a loss function $L$, with the goal of minimizing the loss and training the neural network.

## Usage

```python
import numpy as np
from michigrad.engine import Value
from michigrad.visualize import show_graph

# Weight definition
np.random.seed(42)
W0 = Value(np.random.random(), name='W₀')
W1 = Value(np.random.random(), name='W₁')
b = Value(np.random.random(), name='b')
print(W0)  # prints Value(data=0.3745401188473625, grad=0, name=W₀)

# Training dataset definition
x0 = Value(.5, name="x₀")
x1 = Value(1., name="x₁")
y = Value(2., name="y")

# Forward pass
yhat = x0*W0 + x1*W1 + b
yhat.name = "ŷ"
print(yhat)  # prints Value(data=1.8699783076450025, grad=0, name=ŷ)

L = (y - yhat) ** 2
L.name = "L"
print(L)  # prints Value(data=0.016905640482857615, grad=0, name=L)

# Backward pass
L.backward()

print(L)  # prints Value(data=0.016905640482857615, grad=1, name=L)
print(W0)  # prints Value(data=0.3745401188473625, grad=-0.1300216923549975, name=W₀)

# Update weights in the opposite direction of the gradient
# (Gradient Descent step implementation would go here)

show_graph(L, rankdir="TB", format="png")
