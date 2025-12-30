
import torch
import torch.nn as nn
import torch.optim as optim

def test_xor_pytorch():
    """

    XOR Truth Table:
    x1 | x2 | y
    ---|---|---
    0  | 0  | 0
    0  | 1  | 1
    1  | 0  | 1
    1  | 1  | 0
    """
    
    # Inputs
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], requires_grad=False)
    # Targets
    y = torch.tensor([[0.], [1.], [1.], [0.]], requires_grad=False)

    # MLP comparable to Michigrad:
    # 2 inputs -> 4 hidden neurons (ReLU) -> 1 output (Linear with raw logits or Sigmoid)
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Sigmoid() 
    )

    optimizer = optim.SGD(model.parameters(), lr=0.5)
    criterion = nn.BCELoss() # Binary Cross Entropy

    for epoch in range(500):
        # Forward pass
        y_pred = model(X)
        loss = criterion(y_pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    print("\nFinal Predictions:")
    with torch.no_grad():
        print(model(X))

if __name__ == "__main__":
    test_xor_pytorch()
