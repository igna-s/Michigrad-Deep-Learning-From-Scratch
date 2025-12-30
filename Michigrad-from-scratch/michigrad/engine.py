import math

class Value:
    """
    Stores a single scalar value and its gradient, functioning as a node 
    in a computational graph for backpropagation.
    """

    def __init__(self, data, _children=(), _op='', name=''):
        self.data = data
        self.grad = 0  # Represents the derivative of the output with respect to this node
        self.name = name
        
        # Internal variables for autograd graph construction
        self._backward = lambda: None  # Function to propagate gradients to children
        self._prev = set(_children)    # Set of parent nodes for this operation
        self._op = _op                 # The operation symbol for visualization/debugging

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Addition rule: gradients are distributed equally to both terms (local derivative is 1.0)
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Product rule: derivative is the value of the other node scaled by the upstream gradient
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            # Power rule: d/dx [x^n] = n * x^(n-1)
            self.grad += (other * (self.data**(other - 1))) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0. if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            # ReLU derivative: 1 if x > 0, else 0 (at 0, it's technically undefined, but usually set to 0)
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), f'exp')

        def _backward():
            # Exponential derivative: d/dx [e^x] = e^x
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        """ Hyperbolic tangent activation function """
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # Tanh derivative: d/dx [tanh(x)] = 1 - tanh(x)^2
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out


    # =============================================================================
    # IMPLEMENTATION EXERCISES (MICHIGRAD)
    # =============================================================================

    # 1. Implement the ReLU activation function (easy)
    # Hint: follow the Michigrad reference implementation.

    # 2. Implement the sigmoid activation function (intermediate)
    # Hint: Use these definitions for the function and its derivative:
    #
    # sigmoid(z) = 1 / (1 + exp(-z))
    # sigmoid'(z) = exp(z) / (1 + exp(z))^2
    #
    # Note: Remember that for the backward pass, you need to apply the chain rule.

    # 3. Implement power for Values (difficult)
    # Hint: It should work for Value(3) ** Value(2) 
    # or for expressions like: Value(3) ** (Value(5) + 2)
    # Tip: Make sure to correctly handle the case where the exponent 
    # is not a Value-type object (int or float).


    def relu(self):
            """ Rectified Linear Unit activation function """
            out = Value(0. if self.data < 0 else self.data, (self,), 'ReLU')

            def _backward():
                # The gradient is 1 if the input was positive, 0 otherwise
                self.grad += (out.data > 0) * out.grad
            out._backward = _backward

            return out



    def sigmoid(self):
            """ 
            Sigmoid activation function: 1 / (1 + exp(-x))
            Derivative: sigmoid(x) * (1 - sigmoid(x))
            """
            res = 1 / (1 + math.exp(-self.data))
            out = Value(res, (self,), 'sigmoid')

            def _backward():
                # Using the simplified derivative formula for sigmoid
                # Chain rule: local_derivative * upstream_gradient
                local_derivative = out.data * (1 - out.data)
                self.grad += local_derivative * out.grad
            out._backward = _backward

            return out



    def __pow__(self, other):
            """
            Supports both constant exponents and Value-object exponents.
            Example: a ** 2 or a ** b
            """
            # Ensure 'other' is handled correctly whether it's a Value or a scalar
            is_value = isinstance(other, Value)
            other_data = other.data if is_value else other
            
            out = Value(self.data**other_data, (self, other) if is_value else (self,), f'**{other_data}')

            def _backward():
                # Gradient for the base (Power Rule): d/dx [x^n] = n * x^(n-1)
                # We add a safety check for self.data == 0 if the exponent is less than 1
                if self.data != 0:
                    self.grad += (other_data * (self.data**(other_data - 1))) * out.grad
                
                # Gradient for the exponent (Exponential Rule): d/dy [a^y] = a^y * ln(a)
                # This only applies if 'other' is a Value node in the graph
                if is_value:
                    # Logarithm is only defined for positive bases
                    if self.data > 0:
                        other.grad += (out.data * math.log(self.data)) * out.grad
            
            out._backward = _backward
            return out












    def backward(self):
        """
        Executes backpropagation starting from this node. 
        Constructs a topological sort of the graph to ensure the chain rule 
        is applied in the correct order (from output back to inputs).
        """
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        # Set the base gradient to 1.0 (d_out/d_out = 1.0)
        self.grad = 1.0
        # Apply the chain rule in reverse topological order
        for v in reversed(topo):
            v._backward()

    # --- Utility Methods for Arithmetic Flexibility ---

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, op='{self._op}', name='{self.name}')"



