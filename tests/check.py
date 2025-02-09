from rich.console import Console
from chibigrad.tensor import Tensor
from chibigrad.linear import Linear
import numpy as np
import psutil
import gc

console = Console()

def quick_sanity_check():
    """Quick sanity checks for critical functionality"""
    
    def test_minimal_network():
        """Test basic network functionality"""
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        model = Linear(2, 1)
        y = model(x)
        loss = (y - Tensor([[3.0]])) ** 2
        loss.backward()
        
        assert y.requires_grad, "Gradient tracking failed"
        assert model.weight.grad is not None, "Backprop failed"
        return "✓ Network Test"
    
    def test_memory_footprint():
        """Test memory management"""
        process = psutil.Process()
        initial_mem = process.memory_info().rss / 1024 / 1024
        
        tensors = []
        for _ in range(10):
            t = Tensor(np.random.randn(10, 10), requires_grad=True)
            tensors.append(t)
        
        del tensors
        gc.collect()
        
        final_mem = process.memory_info().rss / 1024 / 1024
        assert (final_mem - initial_mem) < 5, "Memory leak detected"
        return "✓ Memory Test"
    
    def test_numerical_stability():
        """Test handling of extreme values"""
        x1 = Tensor([[1e-7]], requires_grad=True)
        x2 = Tensor([[1e7]], requires_grad=True)
        
        y1 = (x1 ** 2).mean()
        y2 = (x2 ** 2).mean()
        
        y1.backward()
        y2.backward()
        
        assert not np.isnan(x1.grad).any(), "Gradient became NaN"
        assert not np.isinf(x2.grad).any(), "Gradient became Inf"
        return "✓ Stability Test"
    
    def test_broadcasting():
        """Test broadcasting operations"""
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        b = Tensor([0.1, 0.2, 0.3], requires_grad=True)
        y = x + b
        loss = y.mean()
        loss.backward()
        
        assert b.grad.shape == b.data.shape, "Broadcasting gradient shape mismatch"
        return "✓ Broadcasting Test"
    
    # Run all checks
    checks = [
        test_minimal_network,
        test_memory_footprint,
        test_numerical_stability,
        test_broadcasting
    ]
    
    console.print("\n🔍 Running Quick Sanity Checks")
    console.print("─" * 30)
    
    all_passed = True
    for check in checks:
        try:
            result = check()
            console.print(f"{result}")
        except AssertionError as e:
            console.print(f"❌ {check.__name__}: {str(e)}", style="red")
            all_passed = False
        except Exception as e:
            console.print(f"❌ {check.__name__}: Unexpected error: {str(e)}", style="red")
            all_passed = False
    
    console.print("─" * 30)
    
    if all_passed:
        console.print("✨ All checks passed!", style="green")
    else:
        console.print("⚠️ Some checks failed!", style="red")
        return False
    return True

if __name__ == "__main__":
    quick_sanity_check()
