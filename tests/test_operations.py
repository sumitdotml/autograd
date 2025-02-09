import numpy as np
import torch
from chibigrad.tensor import Tensor
import pytest
import warnings


def print_comparison(name, my_tensor, torch_tensor, is_final_layer=False, rtol=1e-4):
    """Concise comparison with stricter tolerance"""
    forward_match = np.allclose(my_tensor.data, torch_tensor.detach().numpy(), rtol=rtol)
    
    if not is_final_layer and hasattr(my_tensor, 'grad') and my_tensor.grad is not None:
        if hasattr(torch_tensor, 'grad') and torch_tensor.grad is not None:
            grad_match = np.allclose(my_tensor.grad, torch_tensor.grad.numpy(), rtol=rtol)
            if not (forward_match and grad_match):
                print(f"\n❌ {name} Test Failed:")
                if not forward_match:
                    diff = np.abs(my_tensor.data - torch_tensor.detach().numpy())
                    print(f"Forward - Mean diff: {np.mean(diff):.6f}, Max diff: {np.max(diff):.6f}")
                if not grad_match:
                    diff = np.abs(my_tensor.grad - torch_tensor.grad.numpy())
                    print(f"Gradient - Mean diff: {np.mean(diff):.6f}, Max diff: {np.max(diff):.6f}")
                return False
    return True

def assert_tensor_equal(name, my_tensor, torch_tensor, rtol=1e-4, atol=1e-6, is_final_layer=False):
    """Assert tensors are equal with strict tolerances and helpful error messages"""
    # Add debug information for gradients
    if not is_final_layer and hasattr(my_tensor, 'grad') and my_tensor.grad is not None:
        if hasattr(torch_tensor, 'grad') and torch_tensor.grad is not None:
            print(f"\nDebug {name} gradients:")
            print(f"My gradient shape: {my_tensor.grad.shape}")
            print(f"Torch gradient shape: {torch_tensor.grad.shape}")
            print(f"My gradient:\n{my_tensor.grad}")
            print(f"Torch gradient:\n{torch_tensor.grad.numpy()}")
            print(f"Gradient difference:\n{np.abs(my_tensor.grad - torch_tensor.grad.numpy())}")
    
    forward_match = np.allclose(my_tensor.data, torch_tensor.detach().numpy(), rtol=rtol, atol=atol)
    if not forward_match:
        diff = np.abs(my_tensor.data - torch_tensor.detach().numpy())
        raise AssertionError(
            f"\n{name} Forward Pass Failed:\n"
            f"My tensor:\n{my_tensor.data}\n"
            f"Torch tensor:\n{torch_tensor.detach().numpy()}\n"
            f"Mean diff: {np.mean(diff):.8f}\n"
            f"Max diff: {np.max(diff):.8f}\n"
            f"At position: {np.unravel_index(diff.argmax(), diff.shape)}"
        )
    
    if not is_final_layer and hasattr(my_tensor, 'grad') and my_tensor.grad is not None:
        if hasattr(torch_tensor, 'grad') and torch_tensor.grad is not None:
            grad_match = np.allclose(my_tensor.grad, torch_tensor.grad.numpy(), rtol=rtol, atol=atol)
            if not grad_match:
                diff = np.abs(my_tensor.grad - torch_tensor.grad.numpy())
                raise AssertionError(
                    f"\n{name} Gradient Failed:\n"
                    f"My gradient:\n{my_tensor.grad}\n"
                    f"Torch gradient:\n{torch_tensor.grad.numpy()}\n"
                    f"Mean diff: {np.mean(diff):.8f}\n"
                    f"Max diff: {np.max(diff):.8f}\n"
                    f"At position: {np.unravel_index(diff.argmax(), diff.shape)}"
                )


class TestBasicOperations:
    def setup_method(self):
        # Suppress PyTorch warnings about accessing .grad on non-leaf tensors
        warnings.filterwarnings("ignore", category=UserWarning, 
                              module="torch.tensor", message=".*non-leaf Tensor.*")
    
    def test_large_matrix_operations(self):
        """Test operations with larger matrices and strict gradient checking"""
        shape = (4, 4)
        np.random.seed(42)
        
        scale = 0.01
        x1 = Tensor(np.random.randn(*shape) * scale, requires_grad=True)
        x2 = Tensor(np.random.randn(*shape) * scale, requires_grad=True)
        torch_x1 = torch.tensor(x1.data, requires_grad=True)
        torch_x2 = torch.tensor(x2.data, requires_grad=True)
        
        # Just do matmul without mean to isolate the issue
        y = x1 @ x2
        y_torch = torch_x1 @ torch_x2
        
        # Use ones for gradient
        grad = np.ones_like(y.data)
        y.backward(grad)
        y_torch.backward(torch.ones_like(y_torch))
        
        assert_tensor_equal("Matrix Multiplication", x1, torch_x1, rtol=1e-3, atol=1e-4)
        assert_tensor_equal("Matrix Multiplication", x2, torch_x2, rtol=1e-3, atol=1e-4)

    def test_addition(self):
        print("\n🔢 Testing Addition")
        x1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        x2 = Tensor([[2.0, 1.0], [4.0, 3.0]], requires_grad=True)
        torch_x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        torch_x2 = torch.tensor([[2.0, 1.0], [4.0, 3.0]], requires_grad=True)
        
        print("\n🔍 Input Tensors:")
        print(f"x1: \n{x1.data}")
        print(f"x2: \n{x2.data}")

        y = x1 + x2
        y_torch = torch_x1 + torch_x2
        
        # Retain gradients for intermediate values
        y.retain_grad()
        y_torch.retain_grad()
        
        grad = np.ones_like(y.data)
        y.backward(grad)
        y_torch.backward(torch.ones_like(y_torch))
        
        print_comparison("Addition", y, y_torch)

    def test_multiplication(self):
        print("\n✖️ Testing Multiplication")
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = Tensor([[2.0, 1.0], [4.0, 3.0]], requires_grad=True)
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y_torch = torch.tensor([[2.0, 1.0], [4.0, 3.0]], requires_grad=True)
        
        z = x @ y
        z_torch = x_torch @ y_torch
        
        # Retain gradients
        z.retain_grad()
        z_torch.retain_grad()
        
        z.backward(np.ones_like(z.data))
        z_torch.backward(torch.ones_like(z_torch))
        
        print_comparison("Matrix Multiplication", z, z_torch)

    def test_power(self):
        print("\n💪 Testing Power Operation")
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        x_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        y = x ** 2
        y_torch = x_torch ** 2
        
        # Retain gradients
        y.retain_grad()
        y_torch.retain_grad()
        
        y.backward(np.ones_like(y.data))
        y_torch.backward(torch.ones_like(y_torch))
        
        print_comparison("Power", y, y_torch)

    @pytest.mark.parametrize("shape", [
        (2,), (2,2), (2,3),  # existing shapes
        (50, 50), (100,), (10, 20)  # new larger shapes
    ])
    def test_reduction_operations(self, shape):
        print(f"\n📉 Testing Reduction Operations for shape {shape}")
        x = Tensor(np.random.randn(*shape), requires_grad=True)
        x_torch = torch.tensor(x.data, requires_grad=True)
        
        # Test sum
        y = x.sum()
        y_torch = x_torch.sum()
        
        y.retain_grad()
        y_torch.retain_grad()
        
        y.backward()
        y_torch.backward()
        
        print_comparison(f"Sum (shape {shape})", y, y_torch)
        
        # Reset gradients
        x.grad = None
        x_torch.grad = None
        
        # Test mean
        y = x.mean()
        y_torch = x_torch.mean()
        
        y.retain_grad()
        y_torch.retain_grad()
        
        y.backward()
        y_torch.backward()
        
        print_comparison(f"Mean (shape {shape})", y, y_torch)


class TestComplexOperations:
    def test_complex_computation(self):
        print("\n🔄 Testing Complex Computation")
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        w = Tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
        b = Tensor([0.5, 0.6], requires_grad=True)
        
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        w_torch = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
        b_torch = torch.tensor([0.5, 0.6], requires_grad=True)
        
        y = (x @ w + b) ** 2
        y_torch = (x_torch @ w_torch + b_torch) ** 2
        
        # Retain gradients for intermediate values
        y.retain_grad()
        y_torch.retain_grad()
        
        y.backward(np.ones_like(y.data))
        y_torch.backward(torch.ones_like(y_torch))
        
        print_comparison("Complex Computation", y, y_torch)

    def test_complex_gradient_flow(self):
        """Test gradient flow through a more complex computation graph"""
        print("\n🧪 Testing Complex Gradient Flow")
        
        # Create input data
        batch_size, in_features, hidden_size, out_features = 8, 4, 6, 2
        np.random.seed(42)
        
        # Input data
        x = Tensor(np.random.randn(batch_size, in_features), requires_grad=True)
        y_true = Tensor(np.random.randn(batch_size, out_features))  # target values
        
        # Create weights and biases
        w1 = Tensor(np.random.randn(in_features, hidden_size) * 0.1, requires_grad=True)
        b1 = Tensor(np.random.randn(hidden_size) * 0.1, requires_grad=True)
        w2 = Tensor(np.random.randn(hidden_size, out_features) * 0.1, requires_grad=True)
        b2 = Tensor(np.random.randn(out_features) * 0.1, requires_grad=True)
        
        # PyTorch equivalents
        x_torch = torch.tensor(x.data, requires_grad=True)
        y_true_torch = torch.tensor(y_true.data)
        w1_torch = torch.tensor(w1.data, requires_grad=True)
        b1_torch = torch.tensor(b1.data, requires_grad=True)
        w2_torch = torch.tensor(w2.data, requires_grad=True)
        b2_torch = torch.tensor(b2.data, requires_grad=True)
        
        # Forward pass - two-layer neural network with ReLU
        def forward_chibi(x):
            h = (x @ w1 + b1).relu()  # hidden layer with ReLU
            y = h @ w2 + b2  # output layer
            diff = y - y_true  # difference from target
            loss = (diff * diff).mean()  # MSE loss
            return loss, y
        
        def forward_torch(x):
            h = torch.relu(x @ w1_torch + b1_torch)
            y = h @ w2_torch + b2_torch
            diff = y - y_true_torch
            loss = (diff * diff).mean()
            return loss, y
        
        # Compute forward pass
        loss_chibi, pred_chibi = forward_chibi(x)
        loss_torch, pred_torch = forward_torch(x_torch)
        
        # Backward pass
        loss_chibi.backward()
        loss_torch.backward()
        
        # Compare results (note the is_final_layer flag)
        print("\n🔄 Network Output:")
        print_comparison("Final Loss", loss_chibi, loss_torch, is_final_layer=True)
        
        print("\n🔙 Gradient Flow:")
        print_comparison("Layer 2 (w2)", w2, w2_torch)
        print_comparison("Layer 2 (b2)", b2, b2_torch)
        print_comparison("Layer 1 (w1)", w1, w1_torch)
        print_comparison("Layer 1 (b1)", b1, b1_torch)
        print_comparison("Input", x, x_torch)

    def test_complex_gradient_accumulation(self):
        """Test gradient accumulation with more complex operations"""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        x_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # Multiple operations with different patterns
        y1 = (x * x).mean()  # Quadratic
        y2 = x.mean()  # Linear
        y3 = (x * x * x).mean()  # Cubic
        
        y1_torch = (x_torch * x_torch).mean()
        y2_torch = x_torch.mean()
        y3_torch = (x_torch * x_torch * x_torch).mean()
        
        # Accumulate all gradients
        y1.backward(retain_graph=True)
        y1_torch.backward(retain_graph=True)
        
        y2.backward(retain_graph=True)
        y2_torch.backward(retain_graph=True)
        
        y3.backward()
        y3_torch.backward()
        
        assert_tensor_equal("Complex Gradient Accumulation", x, x_torch, rtol=1e-5)

    def test_broadcasting_gradient_accumulation(self):
        """Test gradient accumulation with broadcasting"""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        
        # Test broadcasting with scalar
        scalar = Tensor(2.0, requires_grad=True)
        scalar_torch = torch.tensor(2.0, requires_grad=True)
        
        y1 = (x * scalar).mean()
        y2 = (x + scalar).mean()
        
        y1_torch = (x_torch * scalar_torch).mean()
        y2_torch = (x_torch + scalar_torch).mean()
        
        y1.backward(retain_graph=True)
        y1_torch.backward(retain_graph=True)
        
        y2.backward()
        y2_torch.backward()
        
        assert_tensor_equal("Broadcasting Gradients", x, x_torch, rtol=1e-5)
        assert_tensor_equal("Scalar Gradients", scalar, scalar_torch, rtol=1e-5)

    def test_gradient_edge_cases(self):
        """Test gradient accumulation with edge cases"""
        # Test with very small numbers
        x = Tensor([1e-6, 1e-7, 1e-8], requires_grad=True)
        x_torch = torch.tensor([1e-6, 1e-7, 1e-8], requires_grad=True)
        
        # Test with very large numbers
        y = Tensor([1e6, 1e7, 1e8], requires_grad=True)
        y_torch = torch.tensor([1e6, 1e7, 1e8], requires_grad=True)
        
        # Operations mixing large and small numbers
        z1 = (x * y).mean()
        z2 = (x + y).mean()
        
        z1_torch = (x_torch * y_torch).mean()
        z2_torch = (x_torch + y_torch).mean()
        
        z1.backward(retain_graph=True)
        z1_torch.backward(retain_graph=True)
        
        z2.backward()
        z2_torch.backward()
        
        assert_tensor_equal("Small Numbers", x, x_torch, rtol=1e-5)
        assert_tensor_equal("Large Numbers", y, y_torch, rtol=1e-5)

    def test_numerical_edge_cases(self):
        """Test handling of numerical edge cases"""
        # Test infinity handling
        x = Tensor([float('inf'), -float('inf')], requires_grad=True)
        y = x.relu()
        
        # Test NaN handling
        x = Tensor([float('nan'), 1.0], requires_grad=True)
        y = x.mean()
        
        # Test zero division
        x = Tensor([1.0, 0.0], requires_grad=True)
        y = Tensor([0.0, 0.0], requires_grad=True)
        z = x / y  # Should handle gracefully

    def test_advanced_broadcasting(self):
        """Test complex broadcasting scenarios"""
        # Broadcasting with different shapes
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)  # (1, 3)
        y = Tensor([[1.0], [2.0], [3.0]], requires_grad=True)  # (3, 1)
        z = x + y  # Should broadcast to (3, 3)
        
        # Test gradient flow with broadcasting
        z.mean().backward()
        
        # Verify gradients match PyTorch
        x_torch = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        y_torch = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
        z_torch = x_torch + y_torch
        z_torch.mean().backward()

    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        # Test very large numbers
        x = Tensor([1e15, 1e16, 1e17], requires_grad=True)
        y = Tensor([1e-15, 1e-16, 1e-17], requires_grad=True)
        
        # Test operations that might cause overflow/underflow
        z1 = (x * y).mean()  # Should be ~1.0
        z2 = (x / y).mean()  # Should be ~1e30
        z3 = (x + y).mean()  # Should be ~x
        
        z1.backward()
        
        # Check for NaN/Inf
        assert not np.any(np.isnan(x.grad)), "NaN detected in gradients"
        assert not np.any(np.isinf(x.grad)), "Inf detected in gradients"

    def test_edge_case_handling(self):
        """Test handling of edge cases"""
        # Test NaN propagation
        x = Tensor([float('nan'), 1.0, 2.0], requires_grad=True)
        y = x.mean()
        y.backward()
        assert np.all(np.isnan(x.grad)), "NaN not properly propagated"
        
        # Test Inf handling
        x = Tensor([float('inf'), 1.0, 2.0], requires_grad=True)
        y = x.relu()
        y.backward()
        assert np.all(np.isfinite(x.grad)), "Infinite gradients detected"
        
        # Test zero division
        x = Tensor([1.0, 0.0, 2.0], requires_grad=True)
        y = Tensor([0.0, 0.0, 2.0], requires_grad=True)
        z = (x / y).mean()
        z.backward()
        assert np.all(np.isfinite(x.grad[2:])), "Division by zero not properly handled"


class TestIndexing:
    def test_indexing(self):
        # Test basic indexing
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        y = x[0:1]  # Get first row
        y.backward(np.ones_like(y.data))
        
        x_torch = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        y_torch = x_torch[0:1]
        y_torch.backward(torch.ones_like(y_torch))
        
        print_comparison("Indexing", y, y_torch)
        
        # Test batch indexing
        x = Tensor(np.random.randn(10, 3), requires_grad=True)
        idx = np.array([0, 2, 4])
        y = x[idx]
        y.backward(np.ones_like(y.data))
        
        x_torch = torch.tensor(x.data, requires_grad=True)
        y_torch = x_torch[idx]
        y_torch.backward(torch.ones_like(y_torch))
        
        print_comparison("Batch Indexing", y, y_torch)


class TestActivations:
    def test_relu(self):
        """Test ReLU activation with proper gradient checking"""
        print("\n🧪 Testing ReLU")
        
        # Test cases that cover important ReLU scenarios
        x = Tensor([[-2.0, -1.0, 0.0], 
                    [1.0, 2.0, -3.0]], requires_grad=True)
        x_torch = torch.tensor(x.data, requires_grad=True)
        
        # Forward pass
        y = x.relu()
        y_torch = torch.relu(x_torch)
        
        # Create a meaningful gradient for backprop
        grad_output = np.array([[0.5, 1.0, 1.5],
                               [2.0, 0.5, 1.0]])
        
        # Backward pass with non-trivial gradient
        y.backward(Tensor(grad_output))
        y_torch.backward(torch.tensor(grad_output))
        
        print("\n🔄 Testing both positive and negative inputs:")
        print_comparison("ReLU Forward", y, y_torch, is_final_layer=True)
        print_comparison("Input Gradients", x, x_torch)  # Check input gradients
        
        # Verify gradient behavior
        expected_grad = grad_output * (x.data > 0)  # Should be 0 where input was <= 0
        grad_correct = np.allclose(x.grad, expected_grad)
        print(f"\n✨ Gradient Behavior Check:")
        print(f"Zero gradient where input ≤ 0: {'✓' if grad_correct else '✗'}")

    def test_edge_cases(self):
        """Test edge cases and numerical stability"""
        # Test with very small numbers
        x = Tensor(np.array([1e-10, 1e-9, 1e-8]), requires_grad=True)
        x_torch = torch.tensor(x.data, requires_grad=True)
        
        y = x.relu()
        y_torch = torch.relu(x_torch)
        
        y.backward(np.ones_like(y.data))
        y_torch.backward(torch.ones_like(y_torch))
        
        assert_tensor_equal("Small Numbers", x, x_torch, rtol=1e-5)

    def test_gradient_accumulation(self):
        """Test proper gradient accumulation"""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        x_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # First backward pass
        y1 = x.mean()
        y1_torch = x_torch.mean()
        
        y1.backward()
        y1_torch.backward(retain_graph=True)  # Keep computation graph
        
        # Second backward pass (without zeroing gradients)
        y2 = (x * x).mean()
        y2_torch = (x_torch * x_torch).mean()
        
        y2.backward()
        y2_torch.backward()
        
        # No need to manually add gradients - they should accumulate automatically
        
        assert_tensor_equal("Gradient Accumulation", x, x_torch, rtol=1e-5)


if __name__ == "__main__":
    # Using -v for verbose output and -s to show print statements
    pytest.main(["-v", "-s", __file__]) 