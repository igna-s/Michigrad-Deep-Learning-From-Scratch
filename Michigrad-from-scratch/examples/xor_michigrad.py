
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from michigrad.engine import Value
from michigrad.nn import Sequential, Linear, ReLU, Sigmoid, MLP

def test_xor_michigrad():
    print("--- Michigrad XOR Training (Modular) ---")
    
    # Inputs
    X = [
        [Value(0.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(0.0)],
        [Value(1.0), Value(1.0)],
    ]
    # Targets
    y = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]

    # Define the model using the new Sequential container
    model = Sequential([
        Linear(2, 4, nonlin=False),
        ReLU(),
        Linear(4, 1, nonlin=False),
        Sigmoid()
    ])

    print(model)

    # Training Loop
    lr = 0.5
    for epoch in range(500):
        total_loss = Value(0.0)
        
        for k in range(4):
            # Forward pass
            out = model(X[k])
            gt = y[k]

            pred = out
            
            loss = (pred - gt)**2
            total_loss += loss
        
        # Reset gradients
        model.zero_grad()
        
        # Backward pass
        total_loss.backward()
        
        # Update
        for p in model.parameters():
            p.data -= lr * p.grad
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.data:.4f}")

    # Predictions
    print("\nFinal Predictions:")
    for k in range(4):
        pred = model(X[k])
        print(f"Input: {[x.data for x in X[k]]} -> Pred: {pred.data:.4f} (Target: {y[k].data})")

if __name__ == "__main__":
    test_xor_michigrad()
