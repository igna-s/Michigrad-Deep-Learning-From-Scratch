
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import math
import michigrad
from michigrad.engine import Value


def test_sanity_check():
    """
    Verifies the correctness of basic scalar operations and backpropagation.
    
    Compares the results (forward pass and gradients) of a small computational graph
    against PyTorch's autograd engine to ensure numerical accuracy.
    """
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = torch.relu(z) + z * x
    h = torch.relu(z * z)
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    assert ymg.data == ypt.data.item()
    assert xmg.grad == xpt.grad.item()

def test_advanced_activations():
    """
    Verifies recently implemented activation functions: sigmoid, exp, and power.
    
    Ensures that custom implementations of these non-linearities match PyTorch reference values.
    """
    tol = 1e-6

    # --- Test Sigmoid and Exponential functions ---
    a = Value(-1.5)
    b = Value(0.8)
    # f = sigmoid(a) + exp(b)
    f = a.sigmoid() + b.exp()
    f.backward()
    
    a_pt = torch.Tensor([-1.5]).double(); a_pt.requires_grad = True
    b_pt = torch.Tensor([0.8]).double(); b_pt.requires_grad = True
    f_pt = torch.sigmoid(a_pt) + torch.exp(b_pt)
    f_pt.backward()

    assert abs(f.data - f_pt.data.item()) < tol
    assert abs(a.grad - a_pt.grad.item()) < tol
    assert abs(b.grad - b_pt.grad.item()) < tol

    # --- Test Power function with Variable Exponent (Value ** Value) ---
    x = Value(2.0)
    y = Value(3.0)
    g = x ** y  # 2^3
    g.backward()

    x_pt = torch.Tensor([2.0]).double(); x_pt.requires_grad = True
    y_pt = torch.Tensor([3.0]).double(); y_pt.requires_grad = True
    g_pt = x_pt ** y_pt
    g_pt.backward()

    assert abs(g.data - g_pt.data.item()) < tol
    assert abs(x.grad - x_pt.grad.item()) < tol
    assert abs(y.grad - y_pt.grad.item()) < tol



def test_more_ops():
    """
    Integration test with a complex arithmetic expression.
    
    Constructs a deeper graph involving addition, multiplication, power, ReLU, 
    and recent additions (sigmoid, exp) to stress-test the backward pass.
    """
    # Initial parameters
    a_val, b_val = -4.0, 2.0
    
    # Michigrad version
    a = Value(a_val)
    b = Value(b_val)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    h = g.exp()             
    res = h.sigmoid()      
    res.backward()
    amg, bmg, resmg = a, b, res

    # PyTorch version
    a = torch.Tensor([a_val]).double()
    b = torch.Tensor([b_val]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + torch.relu(b + a)
    d = d + 3 * d + torch.relu(b - a)
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    h = torch.exp(g)        
    res = torch.sigmoid(h)  
    res.backward()
    apt, bpt, respt = a, b, res

    tol = 1e-6
    # Gradient and forward pass verification
    assert abs(resmg.data - respt.data.item()) < tol
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

if __name__ == "__main__":
    test_sanity_check()
    test_advanced_activations()
    test_more_ops()
    print("All tests (including sigmoid, exp, and pow) passed successfully.")