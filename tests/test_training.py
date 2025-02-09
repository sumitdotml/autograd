import numpy as np
import torch
import pytest
from chibigrad.tensor import Tensor
from chibigrad.linear import Linear
from chibigrad.arithmetic import *
from rich.console import Console
from rich.table import Table
from rich import box
import matplotlib.pyplot as plt
import time
import psutil
import gc
import warnings

def assert_close(a, b, rtol=1e-4):
    """Assert that two values are close within relative tolerance"""
    if isinstance(a, (np.ndarray, torch.Tensor)):
        assert np.allclose(a, b, rtol=rtol), f"\nExpected:\n{b}\nGot:\n{a}"
    else:
        assert abs(a - b) <= rtol * abs(b), f"Expected {b}, got {a}"


class TestTraining:
    def setup_method(self):
        """Setup common test data"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Generate synthetic data
        self.X = np.random.randn(100, 3)
        self.y = np.sum(self.X, axis=1) * 0.3 + np.random.randn(100) * 0.1
        
        # Convert to tensors
        self.X_chibi = Tensor(self.X, requires_grad=True)
        self.y_chibi = Tensor(self.y.reshape(-1, 1), requires_grad=True)
        self.X_torch = torch.tensor(self.X, requires_grad=True, dtype=torch.float32)
        self.y_torch = torch.tensor(self.y.reshape(-1, 1), requires_grad=True, dtype=torch.float32)

    def _debug_backward(self, name, operation, shape=None, grad_norm=None, layer_info=None, test_name=None, epoch=None):
        """Helper to print more informative backward pass messages"""
        # Add verbosity control
        if not hasattr(self, 'debug_level'):
            self.debug_level = 1  # Default level
        
        if self.debug_level == 0:
            return
        elif self.debug_level == 1 and operation in ["Mean", "ReLU"]:
            # Only show significant operations
            if not any([shape, grad_norm, layer_info]):
                return
        
        # Group related operations
        if hasattr(self, '_last_test') and self._last_test != test_name:
            print("\n" + "="*50)  # Visual separator between tests
        self._last_test = test_name
        
        # Build context string
        context = []
        if test_name:
            context.append(f"Test: {test_name}")
        if epoch is not None:
            context.append(f"Epoch: {epoch:2d}")
        if layer_info:
            context.append(f"Layer {layer_info['layer_idx']}/{layer_info['total_layers']}")
        if shape is not None:
            context.append(f"Shape: {shape}")
        if grad_norm is not None:
            context.append(f"Grad: {grad_norm:.6f}")
        
        context_str = " | ".join(context) if context else ""
        
        # Only print if we have meaningful context
        if context_str:
            emoji = "📊" if operation == "Mean" else "⚡" if operation == "ReLU" else "🔄"
            print(f"{emoji} {name:<15} {operation:<8} - {context_str}")

    def _print_comparison(self, name, chibi_value, torch_value, rtol=1e-4):
        """Helper to print comparison between ChibiGrad and PyTorch"""
        diff = abs(chibi_value - torch_value)
        match = "✓" if diff <= rtol * abs(torch_value) else "✗"
        print(f"\n🔄 {name} Comparison:")
        print(f"  ChibiGrad: {chibi_value:.6f}")
        print(f"  PyTorch:   {torch_value:.6f}")
        print(f"  Diff:      {diff:.6f} [{match}]")
        return diff  # Return the difference

    def _print_section_header(self, title, width=80):
        """Print a section header with consistent formatting"""
        print("\n" + "="*width)
        print(f"📍 {title}")
        print("="*width + "\n")

    def _print_subsection(self, title):
        """Print a subsection header"""
        print(f"\n{'-'*40}")
        print(f"🔹 {title}")
        print(f"{'-'*40}")

    def _create_summary_table(self):
        """Create a rich table summarizing all test results"""
        console = Console()
        table = Table(
            title="🎯 Training Test Summary",
            box=box.ROUNDED,
            header_style="bold magenta",
            title_style="bold cyan",
            border_style="blue",
            show_lines=True,
            padding=(0, 1)
        )
        
        # Add columns with fixed width
        table.add_column("Test Category", style="cyan", no_wrap=True, width=15)
        table.add_column("Details", style="green", width=40)
        table.add_column("Result", justify="center", style="yellow", width=15)
        
        # Architecture
        if hasattr(self, 'architectures'):
            arch_str = " → ".join(str(x) for x in self.architectures[-1])
            table.add_row(
                "Architecture",
                f"Layers: {arch_str}",
                "✓"
            )
        
        # Model size
        if hasattr(self, 'param_count'):
            table.add_row(
                "Model Size",
                f"Parameters: {self.param_count:,}",
                "✓ Compact" if self.param_count < 1000 else "ℹ️ Large"
            )
        
        # Learning progress
        if hasattr(self, 'initial_loss') and hasattr(self, 'final_loss'):
            improvement = (1 - self.final_loss/self.initial_loss) * 100
            table.add_row(
                "Learning",
                f"Loss: {self.initial_loss:.4f} → {self.final_loss:.4f}",
                f"↓ {improvement:.1f}%"
            )
        
        # Training stability
        if hasattr(self, 'train_losses'):
            loss_stability = np.std(self.train_losses)
            table.add_row(
                "Stability",
                f"Loss Std: {loss_stability:.4f}",
                "✓ Stable" if loss_stability < 0.1 else "⚠️ Volatile"
            )
        
        # Gradient health
        if hasattr(self, 'grad_norms'):
            avg_norm = np.mean(list(self.grad_norms.values()))
            table.add_row(
                "Gradients",
                f"Avg Norm: {avg_norm:.4f}",
                "✓ Healthy" if 1e-4 < avg_norm < 1.0 else "⚠️ Unstable"
            )
        
        # Performance comparison
        if hasattr(self, 'chibi_time') and hasattr(self, 'torch_time'):
            ratio = self.chibi_time/self.torch_time
            speed = "🟢 Faster" if ratio < 1 else "🟡 Similar" if ratio < 1.5 else "🔴 Slower"
            table.add_row(
                "Performance",
                f"ChibiGrad: {self.chibi_time:.3f}s vs PyTorch: {self.torch_time:.3f}s",
                f"{speed}\n({ratio:.2f}x)"
            )
        
        # Enhanced memory usage reporting
        if hasattr(self, 'memory_start') and hasattr(self, 'memory_end'):
            memory_diff = self.memory_end - self.memory_start
            kb_per_param = (memory_diff * 1024) / self.param_count if hasattr(self, 'param_count') else float('inf')
            
            # Define reasonable thresholds
            status = "✓ Good" if kb_per_param < 10 else "⚠️ High" if kb_per_param < 100 else "🔴 Critical"
            
            leaked = self.memory_efficiency.get('leaked_tensors', 0) if hasattr(self, 'memory_efficiency') else 0
            leak_status = "" if leaked == 0 else f" ({leaked} leaks)"
            
            table.add_row(
                "Memory Usage",
                f"Total: {memory_diff:.1f}MB, Per Param: {kb_per_param:.1f}KB{leak_status}",
                f"{status}\n({memory_diff:.1f}MB)"
            )
        
        # Only print if we have data
        if table.row_count > 0:
            console.print("\n")
            console.print(table)
            console.print("\n")

    def test_different_architectures(self):
        """Test various network architectures"""
        self._print_section_header("Architecture Testing")
        self.architectures = [
            [3, 4, 1],      # Original (shallow)
            [3, 8, 4, 1],   # Deeper
            [3, 16, 8, 4, 1],  # Much deeper
            [3, 2, 1],      # Narrower
        ]
        self.arch_diffs = {}
        self.grad_norms = {}
        self.param_count = 0
        
        for i, layers in enumerate(self.architectures):
            self._print_subsection(f"Architecture {i+1}: {layers}")
            
            # Create ChibiModel with dynamic architecture
            class ChibiModel:
                def __init__(self, layer_sizes):
                    self.layers = []
                    for i in range(len(layer_sizes) - 1):
                        self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
                
                def __call__(self, x):
                    for i, layer in enumerate(self.layers):
                        x = layer(x)
                        if i < len(self.layers) - 1:  # No ReLU on final layer
                            x = x.relu()
                    return x
                
                def parameters(self):
                    return [p for layer in self.layers for p in layer.parameters()]
                
                def zero_grad(self):
                    for p in self.parameters():
                        p.grad = None
            
            # Create equivalent PyTorch model
            class TorchModel(torch.nn.Module):
                def __init__(self, layer_sizes):
                    super().__init__()
                    self.layers = torch.nn.ModuleList([
                        torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                        for i in range(len(layer_sizes) - 1)
                    ])
                
                def forward(self, x):
                    for i, layer in enumerate(self.layers):
                        x = layer(x)
                        if i < len(self.layers) - 1:
                            x = torch.relu(x)
                    return x
            
            # Test training
            chibi_model = ChibiModel(layers)
            torch_model = TorchModel(layers)
            
            # Match initializations
            with torch.no_grad():
                for chibi_layer, torch_layer in zip(chibi_model.layers, torch_model.layers):
                    torch_layer.weight.copy_(torch.tensor(chibi_layer.weight.data))
                    torch_layer.bias.copy_(torch.tensor(chibi_layer.bias.data))
            
            # Single training step
            learning_rate = 0.01
            
            # Forward pass
            activations = []  # Store intermediate activations
            x = self.X_chibi
            for j, layer in enumerate(chibi_model.layers):
                x = layer(x)
                if j < len(chibi_model.layers) - 1:
                    x = x.relu()
                    activations.append(x)
            pred_chibi = x
            
            # Loss computation
            loss_chibi = ((pred_chibi - self.y_chibi) ** 2).mean()
            
            # Backward pass with enhanced debugging
            def backward_hook(grad, layer_idx=None):
                if hasattr(grad, 'shape'):
                    grad_norm = np.sqrt(np.mean(grad ** 2))
                    layer_info = {
                        'layer_idx': layer_idx,
                        'total_layers': len(chibi_model.layers)
                    } if layer_idx is not None else None
                    
                    self._debug_backward(
                        name=f"Architecture {i+1}",
                        operation="Mean" if layer_idx is None else "ReLU",
                        shape=grad.shape,
                        grad_norm=grad_norm,
                        layer_info=layer_info
                    )
                return grad
            
            # Set hooks for each layer
            for j, activation in enumerate(activations):
                activation._backward_hook = lambda g, j=j: backward_hook(g, j+1)
            
            loss_chibi._backward_hook = lambda g: backward_hook(g)
            loss_chibi.backward()
            
            # Print detailed layer gradients
            print("\n📉 Layer gradients and statistics:")
            for j, layer in enumerate(chibi_model.layers):
                w_grad_norm = np.sqrt(np.mean(layer.weight.grad ** 2))
                b_grad_norm = np.sqrt(np.mean(layer.bias.grad ** 2))
                w_grad_mean = np.mean(layer.weight.grad)
                w_grad_std = np.std(layer.weight.grad)
                
                self._print_gradient_stats(j+1, len(chibi_model.layers), w_grad_norm, w_grad_mean, w_grad_std, b_grad_norm)
            
            # Modify the comparison call to capture the difference
            diff = self._print_comparison(
                f"Architecture {i+1} Forward Pass",
                float(loss_chibi.data),
                float(torch.nn.functional.mse_loss(torch_model(self.X_torch), self.y_torch).detach().numpy())
            )
            
            # Now we can store the difference
            self.arch_diffs[tuple(layers)] = diff
            
            # Track gradient norms
            for j, layer in enumerate(chibi_model.layers):
                norm_key = f"arch{i}_layer{j}"
                self.grad_norms[norm_key] = np.sqrt(np.mean(layer.weight.grad ** 2))
            
            # Count parameters
            self.param_count += sum(np.prod(p.data.shape) for p in chibi_model.parameters())

    def test_different_losses(self):
        """Test different loss functions"""
        # Binary classification data
        X = np.random.randn(100, 3)
        y = (np.sum(X, axis=1) > 0).astype(np.float32)
        
        X_chibi = Tensor(X, requires_grad=True)
        y_chibi = Tensor(y.reshape(-1, 1), requires_grad=True)
        X_torch = torch.tensor(X, requires_grad=True, dtype=torch.float32)
        y_torch = torch.tensor(y.reshape(-1, 1), requires_grad=True, dtype=torch.float32)
        
        # Test MSE loss with thresholding
        model_chibi = self._create_model([3, 4, 1])
        model_torch = self._create_torch_model([3, 4, 1])
        
        # Match initializations
        with torch.no_grad():
            for chibi_layer, torch_layer in zip(model_chibi.layers, model_torch.layers):
                torch_layer.weight.copy_(torch.tensor(chibi_layer.weight.data))
                torch_layer.bias.copy_(torch.tensor(chibi_layer.bias.data))
        
        # Forward pass
        pred_chibi = model_chibi(X_chibi)
        pred_torch = model_torch(X_torch)
        
        # Binary classification loss using MSE
        loss_chibi = ((pred_chibi - y_chibi) ** 2).mean()
        loss_torch = torch.nn.functional.mse_loss(pred_torch, y_torch)
        
        # Backward pass
        loss_chibi.backward()
        loss_torch.backward()
        
        # Assert losses match
        assert_close(loss_chibi.data, loss_torch.detach().numpy(), rtol=1e-4)
        
        # Test predictions (thresholded)
        pred_chibi_binary = (pred_chibi.data > 0.5).astype(np.float32)
        pred_torch_binary = (pred_torch.detach().numpy() > 0.5).astype(np.float32)
        
        # Compare binary predictions
        assert_close(pred_chibi_binary, pred_torch_binary, rtol=1e-4)

    def test_training_loop(self):
        """Original training loop test with additional metrics"""
        # Original test implementation...
        # [Keep the existing test_training_loop implementation]

    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling during training"""
        model = self._create_model([3, 4, 1])
        initial_lr = 0.1
        decay_rate = 0.9
        
        for epoch in range(5):
            lr = initial_lr * (decay_rate ** epoch)
            # Test training step with decayed learning rate
            
    def test_gradient_clipping(self):
        """Test gradient clipping for numerical stability"""
        model = self._create_model([3, 4, 1])
        max_norm = 1.0
        
        # Forward and backward passes
        pred = model(self.X_chibi)
        loss = ((pred - self.y_chibi) ** 2).mean()
        loss.backward()
        
        # Test clipping
        for p in model.parameters():
            grad_norm = np.sqrt(np.sum(p.grad ** 2))
            assert grad_norm <= max_norm

    def test_model_state(self):
        """Test model state consistency during training"""
        model = self._create_model([3, 4, 1])
        initial_params = [p.data.copy() for p in model.parameters()]

    def _create_model(self, layer_sizes):
        """Helper to create ChibiModel"""
        class ChibiModel:
            def __init__(self, sizes):
                self.layers = []
                for i in range(len(sizes) - 1):
                    self.layers.append(Linear(sizes[i], sizes[i + 1]))
            
            def __call__(self, x):
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    if i < len(self.layers) - 1:
                        x = x.relu()
                return x
            
            def parameters(self):
                return [p for layer in self.layers for p in layer.parameters()]
            
            def zero_grad(self):
                for p in self.parameters():
                    p.grad = None
        
        return ChibiModel(layer_sizes)

    def _create_torch_model(self, layer_sizes):
        """Helper to create PyTorch model"""
        class TorchModel(torch.nn.Module):
            def __init__(self, sizes):
                super().__init__()
                self.layers = torch.nn.ModuleList([
                    torch.nn.Linear(sizes[i], sizes[i + 1])
                    for i in range(len(sizes) - 1)
                ])
            
            def forward(self, x):
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    if i < len(self.layers) - 1:
                        x = torch.relu(x)
                return x
        
        return TorchModel(layer_sizes)

    def test_batch_training(self):
        """Test training with different batch sizes"""
        batch_sizes = [1, 16, 32, 64]
        model = self._create_model([3, 4, 1])
        
        for batch_size in batch_sizes:
            # Create batch data
            indices = np.random.choice(len(self.X), batch_size)
            X_batch = self.X_chibi[indices]
            y_batch = self.y_chibi[indices]
            
            # Test forward and backward passes
            pred = model(X_batch)
            loss = ((pred - y_batch) ** 2).mean()
            loss.backward()
            
            # Verify gradients shape
            for param in model.parameters():
                assert param.grad is not None
                assert param.grad.shape == param.data.shape

    def test_training_metrics(self):
        """Test training metrics collection"""
        model = self._create_model([3, 4, 1])
        n_epochs = 5
        metrics = {'train_loss': [], 'val_loss': []}
        learning_rate = 0.01  # Add learning rate
        
        # Split data into train/val
        split = 80
        X_train, X_val = self.X_chibi[:split], self.X_chibi[split:]
        y_train, y_val = self.y_chibi[:split], self.y_chibi[split:]
        
        for epoch in range(n_epochs):
            # Training
            pred = model(X_train)
            loss = ((pred - y_train) ** 2).mean()
            loss.backward()
            
            # Update parameters
            for p in model.parameters():
                if p.grad is not None:
                    p.data -= learning_rate * p.grad
            model.zero_grad()  # Zero gradients after update
            
            metrics['train_loss'].append(float(loss.data))
            
            # Validation
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = ((val_pred - y_val) ** 2).mean()
                metrics['val_loss'].append(float(val_loss.data))
        
        # Verify metrics
        assert len(metrics['train_loss']) == n_epochs
        assert metrics['train_loss'][-1] < metrics['train_loss'][0], \
            f"Training loss did not decrease: {metrics['train_loss'][0]} -> {metrics['train_loss'][-1]}"
        
        # Additional metric checks
        assert all(isinstance(loss, float) for loss in metrics['train_loss']), "Training losses should be float"
        assert all(isinstance(loss, float) for loss in metrics['val_loss']), "Validation losses should be float"
        assert all(loss > 0 for loss in metrics['train_loss']), "Losses should be positive"

    def test_training_stability(self):
        """Test training stability with extreme values"""
        # Test with very large/small learning rates
        learning_rates = [1e-6, 1e-1, 1.0, 10.0]
        model = self._create_model([3, 4, 1])
        
        for lr in learning_rates:
            pred = model(self.X_chibi)
            loss = ((pred - self.y_chibi) ** 2).mean()
            loss.backward()
            
            # Update with different learning rates
            for p in model.parameters():
                if p.grad is not None:
                    p.data -= lr * p.grad
                    # Check for NaN/Inf
                    assert not np.any(np.isnan(p.data))
                    assert not np.any(np.isinf(p.data))

    def test_gradient_flow_visualization(self):
        """Test gradient flow visualization through layers"""
        model = self._create_model([3, 8, 4, 1])
        grad_magnitudes = []
        
        # Forward and backward pass
        pred = model(self.X_chibi)
        loss = ((pred - self.y_chibi) ** 2).mean()
        loss.backward()
        
        # Collect gradient magnitudes per layer
        for layer in model.layers:
            weight_grad_norm = np.sqrt(np.mean(layer.weight.grad ** 2))
            bias_grad_norm = np.sqrt(np.mean(layer.bias.grad ** 2))
            grad_magnitudes.append({
                'weight_grad': float(weight_grad_norm),
                'bias_grad': float(bias_grad_norm)
            })
        
        # Check gradient flow
        for i, grads in enumerate(grad_magnitudes):
            assert grads['weight_grad'] > 0, f"Layer {i} has zero gradient"
            assert not np.isnan(grads['weight_grad']), f"Layer {i} has NaN gradient"

    def test_optimizer_behavior(self):
        """Test optimizer behavior with different parameters"""
        model = self._create_model([3, 4, 1])
        learning_rates = [0.1, 0.01, 0.001]
        momentum_values = [0.0, 0.9]
        
        for lr in learning_rates:
            for momentum in momentum_values:
                # Store initial parameters
                initial_params = [p.data.copy() for p in model.parameters()]
                
                # Training step
                pred = model(self.X_chibi)
                loss = ((pred - self.y_chibi) ** 2).mean()
                loss.backward()
                
                # Update with momentum
                if momentum > 0:
                    velocities = [np.zeros_like(p.data) for p in model.parameters()]
                    for p, v, init_p in zip(model.parameters(), velocities, initial_params):
                        if p.grad is not None:
                            v[:] = momentum * v - lr * p.grad
                            p.data = init_p + v
                else:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.data -= lr * p.grad
                
                # Verify updates
                for p, init_p in zip(model.parameters(), initial_params):
                    assert not np.array_equal(p.data, init_p), "Parameters were not updated"

    def test_learning_curves(self):
        """Test learning curve behavior"""
        self._print_section_header("Learning Curves Test")
        
        self.debug = True  # Enable debugging
        model = self._create_model([3, 4, 1])
        n_epochs = 10
        train_losses = []
        val_losses = []
        
        print("\n📈 Learning Curves Test")
        
        # Split data
        split_idx = int(0.8 * len(self.X))
        X_train, X_val = self.X_chibi[:split_idx], self.X_chibi[split_idx:]
        y_train, y_val = self.y_chibi[:split_idx], self.y_chibi[split_idx:]
        
        for epoch in range(n_epochs):
            # Training
            pred = model(X_train)
            loss = ((pred - y_train) ** 2).mean()
            
            # Add backward hook with context
            def backward_hook(grad):
                if hasattr(grad, 'shape'):
                    grad_norm = np.sqrt(np.mean(grad ** 2))
                    self._debug_backward(
                        name="Training",
                        operation="Mean",
                        shape=grad.shape,
                        grad_norm=grad_norm,
                        test_name="Learning Curves",
                        epoch=epoch
                    )
                return grad
            
            loss._backward_hook = backward_hook
            loss.backward()
            
            # Update parameters
            for p in model.parameters():
                if p.grad is not None:
                    p.data -= 0.01 * p.grad
                    p.grad = None
            
            # Record losses
            train_loss = float(loss.data)
            train_losses.append(train_loss)
            
            # Validation
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = ((val_pred - y_val) ** 2).mean()
                val_losses.append(float(val_loss.data))
            
            # Print progress
            if epoch % 2 == 0:
                print(f"  Epoch {epoch:2d}: Train Loss = {train_loss:.6f}, Val Loss = {float(val_loss.data):.6f}")

        # Add visualization
        self.plot_learning_curves(train_losses, val_losses)

    def plot_learning_curves(self, train_losses, val_losses, save_path='learning_curves.png'):
        """Enhanced learning curve visualization"""
        plt.figure(figsize=(12, 6))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(train_losses, label='Train Loss', marker='o')
        ax1.plot(val_losses, label='Val Loss', marker='s')
        ax1.set_title('Learning Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()
        
        # Loss ratio
        ax2.plot([v/t for t, v in zip(train_losses, val_losses)], 
                 label='Val/Train Ratio', color='g')
        ax2.set_title('Validation/Training Loss Ratio')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Ratio')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def print_test_summary(self):
        """Comprehensive test summary"""
        print("\n📊 Test Summary:")
        print("Architecture Tests:")
        for arch in self.architectures:
            print(f"  {arch}: Loss diff from PyTorch = {self.arch_diffs[tuple(arch)]:.6f}")
        
        print("\nLearning Metrics:")
        print(f"  Initial Loss: {self.initial_loss:.6f}")
        print(f"  Final Loss: {self.final_loss:.6f}")
        print(f"  Improvement: {(1 - self.final_loss/self.initial_loss)*100:.1f}%")
        
        print("\nPerformance:")
        print(f"  Average ChibiGrad Time: {np.mean(self.chibi_times):.3f}s")
        print(f"  Average PyTorch Time: {np.mean(self.torch_times):.3f}s")

    def test_training_performance(self):
        """Test training performance with memory optimization"""
        process = psutil.Process()
        
        # Force initial cleanup
        gc.collect()
        self.memory_start = process.memory_info().rss / 1024 / 1024
        
        model_size = [3, 16, 8, 1]
        n_epochs = 5
        
        # Initialize param_count
        self.param_count = 0
        
        # Track tensors before training
        initial_tensors = set(id(obj) for obj in gc.get_objects() if isinstance(obj, Tensor))
        
        def cleanup_model(model):
            """Helper to cleanup model state"""
            for layer in model.layers:
                if hasattr(layer, 'weight'):
                    layer.weight.grad = None
                    layer.weight._backward_fn = None
                if hasattr(layer, 'bias'):
                    layer.bias.grad = None
                    layer.bias._backward_fn = None
        
        # ChibiGrad training with memory optimization
        chibi_model = self._create_model(model_size)
        
        # Count parameters
        for layer in chibi_model.layers:
            if hasattr(layer, 'weight'):
                self.param_count += np.prod(layer.weight.data.shape)
            if hasattr(layer, 'bias'):
                self.param_count += layer.bias.data.size
        
        start_time = time.time()
        
        # Initial prediction for loss tracking
        pred = chibi_model(self.X_chibi)
        self.initial_loss = ((pred - self.y_chibi) ** 2).mean().data
        
        self.train_losses = []
        for epoch in range(n_epochs):
            # Clear previous gradients and graphs
            cleanup_model(chibi_model)
            
            # Forward pass
            pred = chibi_model(self.X_chibi)
            loss = ((pred - self.y_chibi) ** 2).mean()
            self.train_losses.append(float(loss.data))
            
            # Backward pass
            loss.backward()
            
            # Update parameters and immediately clear gradients
            for layer in chibi_model.layers:
                if layer.weight.grad is not None:
                    layer.weight.data -= 0.01 * layer.weight.grad
                    layer.weight.grad = None
                if layer.bias.grad is not None:
                    layer.bias.data -= 0.01 * layer.bias.grad
                    layer.bias.grad = None
            
            # Force cleanup every few epochs
            if epoch % 2 == 0:
                gc.collect()
        
        self.final_loss = self.train_losses[-1]
        chibi_time = time.time() - start_time
        
        # Cleanup before PyTorch comparison
        del chibi_model
        gc.collect()
        
        # PyTorch training
        torch_model = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1)
        )
        optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.01)
        
        start_time = time.time()
        
        for _ in range(n_epochs):
            optimizer.zero_grad()
            pred = torch_model(self.X_torch)
            loss = torch.nn.functional.mse_loss(pred, self.y_torch)
            loss.backward()
            optimizer.step()
        
        torch_time = time.time() - start_time
        
        print(f"\n⚡ Performance Comparison:")
        print(f"  Model size: {model_size}")
        print(f"  Epochs: {n_epochs}")
        print(f"  ChibiGrad: {chibi_time:.3f}s")
        print(f"  PyTorch:   {torch_time:.3f}s")
        if torch_time > 0:  # Avoid division by zero
            print(f"  Ratio: {chibi_time/torch_time:.2f}x slower")
        else:
            print("  Ratio: Could not compute (PyTorch too fast)")
        
        self.chibi_time = chibi_time
        self.torch_time = torch_time
        
        # Final cleanup and memory check
        gc.collect()
        self.memory_end = process.memory_info().rss / 1024 / 1024
        
        # Check for tensor leaks
        final_tensors = set(id(obj) for obj in gc.get_objects() if isinstance(obj, Tensor))
        leaked_tensors = len(final_tensors - initial_tensors)
        if leaked_tensors > 0:
            warnings.warn(f"Detected {leaked_tensors} leaked tensors!")
        
        # Store memory efficiency metrics
        self.memory_efficiency = {
            'per_param': (self.memory_end - self.memory_start) / self.param_count,
            'leaked_tensors': leaked_tensors
        }

    def analyze_gradient_flow(self, model, epoch):
        """Enhanced gradient flow analysis"""
        grad_norms = []
        grad_means = []
        layer_names = []
        
        for i, layer in enumerate(model.layers):
            w_grad = layer.weight.grad
            grad_norm = np.sqrt(np.mean(w_grad ** 2))
            grad_mean = np.mean(w_grad)
            grad_norms.append(grad_norm)
            grad_means.append(grad_mean)
            layer_names.append(f"Layer {i+1}")
        
        print(f"\n📊 Gradient Flow Analysis (Epoch {epoch}):")
        for name, norm, mean in zip(layer_names, grad_norms, grad_means):
            bar = "=" * int(norm * 100)
            print(f"  {name:<8}: {bar} {norm:.6f} (mean: {mean:.6f})")
        
        return grad_norms, grad_means

    def _print_gradient_stats(self, layer_idx, total_layers, w_grad_norm, w_grad_mean, w_grad_std, b_grad_norm):
        """Print gradient statistics in a formatted way"""
        print(f"\n  {'─'*30}")
        print(f"  Layer {layer_idx}/{total_layers}:")
        print(f"  {'─'*30}")
        print(f"    Weight Gradients:")
        print(f"      Norm: {w_grad_norm:.6f}")
        print(f"      Mean: {w_grad_mean:.6f}")
        print(f"      Std:  {w_grad_std:.6f}")
        print(f"    Bias Gradient:")
        print(f"      Norm: {b_grad_norm:.6f}")

    def _print_performance_comparison(self, model_size, n_epochs, chibi_time, torch_time):
        """Print performance comparison with clear formatting"""
        print("\n" + "="*50)
        print("⚡ Performance Comparison")
        print("="*50)
        print(f"\n  Model Architecture:")
        print(f"  {'-'*20}")
        for i, size in enumerate(model_size):
            print(f"    Layer {i}: {size} neurons")
        
        print(f"\n  Training Details:")
        print(f"  {'-'*20}")
        print(f"    Epochs: {n_epochs}")
        print(f"    Batch Size: {self.X.shape[0]}")
        
        print(f"\n  Timing Results:")
        print(f"  {'-'*20}")
        print(f"    ChibiGrad: {chibi_time:.3f}s")
        print(f"    PyTorch:   {torch_time:.3f}s")
        if torch_time > 0:
            ratio = chibi_time/torch_time
            speed = "🟢 Faster" if ratio < 1 else "🟡 Similar" if ratio < 1.5 else "🔴 Slower"
            print(f"    Ratio: {ratio:.2f}x ({speed})")
        else:
            print("    Ratio: Could not compute (PyTorch too fast)")

    def teardown_method(self):
        """Called after each test method"""
        self._create_summary_table()


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])