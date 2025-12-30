import random
from michigrad.engine import Value


# =============================================================================
# IMPLEMENTATION EXERCISES (MICHIGRAD)
# =============================================================================


# --- Michigrad / Micrograd Exercises ---

# 1. Implement the 'bias' parameter in Neuron and Layer to allow creating neurons without bias. (Easy)
# Hint: Check the Michigrad implementation.


# 2. Implement the XOR model using PyTorch. (Easy if you've used PyTorch before, Intermediate otherwise)
# Hint: PyTorch doesn't implement an MLP directly. You can redefine the MLP class 
# using torch.Linear instead of manual neuron layers (Layer and Neuron).


# 3. Implement activation functions as layers. (Intermediate)
# Hint: Replace 'Layer' with 'Linear', and define a class for each activation function. 
# The Linear layer will act as a layer of neurons, and the activation layer will apply 
# the activation function to every output of all neurons from the previous layer.


# 4. Implement the Module class that allows creating models as lists of modules. 
# All modules must support the __call__(self, x) function which performs the forward pass. (Difficult)
# Hint: You should be able to create a model as a list of modules, like this:
# model = Model([Linear(2, 4), Linear(4, 4, bias=False), Linear(4, 3), Tanh(3)])
# model(x)










class Module:
    """
    Base class for all neural network modules.
    
    Provides standard methods for parameter management and gradient zeroing.
    Subclasses should implement the `forward` logic (via `__call__`) and `parameters`.
    """

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True, bias=True):
        """
        Initializes a neuron with random weights and zero bias.
        
        Args:
            nin: Number of input connections (dimensionality of input vector).
            nonlin: If True, applies ReLU activation. If False, remains linear.
        """
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0) if bias else None
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), Value(0))
        if self.b:
           act += self.b
        return act.relu() if self.nonlin else act


    def parameters(self):
        return self.w + ([self.b] if self.b else [])

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Linear(Module):
    """
    A fully connected layer.
    
    Performs the operation: y = x * W^T + b
    Previously known as 'Layer'.
    """

    def __init__(self, nin, nout, **kwargs):
        """
        Args:
            nin: Number of input features
            nout: Number of output neurons
        """
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Linear({len(self.neurons[0].w)}, {len(self.neurons)})"


# --- Activation Layers ---

class ReLU(Module):
    """
    Applies the rectified linear unit function element-wise: ReLU(x) = max(0, x)
    """
    def __call__(self, x):
        # Handle list of Values (multi-neuron output)
        if isinstance(x, list):
            return [val.relu() for val in x]
        # Handle single Value (single neuron output)
        return x.relu()

    def __repr__(self):
        return "ReLU()"

class Tanh(Module):
    """
    Applies the Hyperbolic Tangent (Tanh) function element-wise.
    """
    def __call__(self, x):
        if isinstance(x, list):
            return [val.tanh() for val in x]
        return x.tanh()

    def __repr__(self):
        return "Tanh()"

class Sigmoid(Module):
    """
    Applies the Sigmoid function element-wise.
    """
    def __call__(self, x):
        if isinstance(x, list):
            return [val.sigmoid() for val in x]
        return x.sigmoid()

    def __repr__(self):
        return "Sigmoid()"




class Sequential(Module):
    """
    A sequential container for piling modules.
    
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an `add` method could be implemented, but list in __init__ is standard.
    """
    
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        layers_str = ", ".join(str(layer) for layer in self.layers)
        return f"Sequential([{layers_str}])"


# Deprecated MLP implementation (kept for backward compatibility, but refactored to use new structure logic if desired)
# For this exercise, we'll keep the old one but note it's less flexible than Sequential.
# Or better: Re-implement MLP to return a Sequential model!

class MLP(Module):
    """
    Multi-Layer Perceptron.
    
    Refactored to be a wrapper around Sequential for ease of use.
    """

    def __init__(self, nin, nouts):
        """
        Args:
            nin: Number of inputs.
            nouts: List of integers defining the size of each subsequent layer.
        """
        sz = [nin] + nouts
        layers = []
        for i in range(len(nouts)):
            layers.append(Linear(sz[i], sz[i+1], nonlin=i!=len(nouts)-1)) # Using legacy nonlin for compatibility
        self.model = Sequential(layers)

    def __call__(self, x):
        return self.model(x)

    def parameters(self):
        return self.model.parameters()

    def __repr__(self):
        return f"MLP({self.model})"




        