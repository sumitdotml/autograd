import torch


def verify_gradient():
    # Create tensors with requires_grad=True to track gradients
    a = torch.tensor(2.0, requires_grad=True)

    # Recreate the same computation graph
    b = a * 3.0
    c = a / 3.0
    d = 4 * b * c
    e = d - a

    # Compute backward pass
    e.backward()

    print("PyTorch computed gradient for 'a':", a.grad.item())
    print("My manual calculation:", 15.0)
    print("Match:", abs(a.grad.item() - 15.0) < 1e-6)

    # Print intermediate values for debugging
    print("\nIntermediate values:")
    print(f"b = {b.item()}")
    print(f"c = {c.item()}")
    print(f"d = {d.item()}")
    print(f"e = {e.item()}")


if __name__ == "__main__":
    verify_gradient()
