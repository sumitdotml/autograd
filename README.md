# ChibiGrad 🐤

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A miniature autograd engine designed for educational purposes, implementing automatic differentiation in pure NumPy.

> [!NOTE]
> __Status:__ Active Development  
> Not production-ready - created for learning purposes

---

## Why "ChibiGrad"?

"Chibi" (ちび) means "small" or "miniature" in Japanese. This is a tiny autograd engine designed for learning purposes - hence ChibiGrad!

---

## Features

- 🧮 Tensor operations with automatic differentiation
- 📈 Neural network layers (Linear)
- 🔧 Basic operations (Add, Multiply, Power, etc.)
- 📊 Loss functions (MSE) (more to come)
- 🔄 GPU-free (NumPy based)
- ✅ PyTorch-like API for easier learning

---

## Installation

### For Basic Usage

chibigrad requires Python 3.8+ and NumPy. Clone and install dependencies:

1. **Clone the Repository**
```bash
git clone https://github.com/sumitdotml/chibigrad.git
cd chibigrad
```

2. **Create Virtual Environment**
```bash
python -m venv .venv         # Create virtual environment
source .venv/bin/activate   # Activate (Linux/Mac)
.venv\Scripts\activate      # Activate (Windows)
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### For Package Development

1. **Clone and Set Up Virtual Environment**
```bash
python -m venv .venv         # Create virtual environment
source .venv/bin/activate   # Activate (Linux/Mac)
.venv\Scripts\activate      # Activate (Windows)
```

2. **Editable Install with Development Dependencies**
```bash
pip install -e ".[tests]"    # Install package in editable mode with test deps
```

3. **Verify Installation**
```bash
python -c "import chibigrad; print(chibigrad.__version__)"
# Should output: 0.1.0
```

4. **Development Workflow**
```bash
# Install pre-commit hooks (optional but recommended)
pre-commit install

# Run tests after changes
python -m tests.check --test all

# Reinstall after major changes
pip install -e . --force-reinstall
```

### Dependency Management

| Dependency Group | Packages                          | Purpose                     |
|------------------|-----------------------------------|----------------------------|
| core             | `numpy`, `rich`                  | Core functionality         |
| tests            | `torch`                          | Gradient comparison tests  |
| dev              | `black`, `flake8`, `mypy`        | Code quality (optional)     |

To add development dependencies:
```bash
pip install black flake8 mypy  # Code formatting and linting
```

---

## Basic Usage

### Tensor Operations
```python
from chibigrad.tensor import Tensor
import numpy as np

# Create tensors
x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = Tensor([[2.0, 1.0], [4.0, 3.0]], requires_grad=True)

# Basic arithmetic
z = x + y  # Addition
w = x * y  # Element-wise multiplication
m = x @ y  # Matrix multiplication

# Reduction operations
mean = x.mean()
sum_x = x.sum()

# Activation functions
activated = x.relu()  # ReLU activation

# Backward pass
loss = (z ** 2).mean()
loss.backward()

# Access gradients
print(x.grad)  # Gradients for x
```

### Neural Network Example
```python
from chibigrad.tensor import Tensor
from chibigrad.linear import Linear
from chibigrad.loss import MSELoss

# Create model
class SimpleNN:
    def __init__(self):
        self.linear1 = Linear(2, 4)
        self.linear2 = Linear(4, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = x.relu()
        return self.linear2(x)

# Training data
X = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y_true = Tensor([[3.0], [7.0]])

# Model and loss
model = SimpleNN()
criterion = MSELoss()

# Forward pass
y_pred = model.forward(X)
loss = criterion(y_pred, y_true)

# Backward pass
loss.backward()
```

---

## Testing

Runs essential smoke tests:
- Basic network functionality
- Memory management
- Numerical stability
- Broadcasting operations

### Comprehensive Test Suite
```bash
# Run specific test files
python -m tests.test_operations  # Basic operations
python -m tests.test_training    # Training & optimization
python -m tests.check            # Sanity checks

# Run all tests
python -m pytest tests/
```

---

## Project Structure and Documentation

```
chibigrad/
├── chibigrad/ # Core autograd engine implementation
│ ├── tensor.py # Tensor class with automatic differentiation
│ ├── operation.py # Base class for all operations
│ ├── arithmetic.py # Arithmetic operations (Add, Multiply, etc.)
│ ├── matmul.py # Matrix multiplication operation
│ ├── linear.py # Neural network Linear layer
│ ├── loss.py # Loss functions (MSE currently)
│ ├── activations.py # Activation functions (Placeholder for now, will be added soon)
│ ├── optim.py # Optimizers (Placeholder for now, will be added soon)
│ └── module.py # Base class for neural network modules
├── tests/ # Comprehensive test suite
│ └── check.py # Gradient comparison tests against PyTorch
├── requirements.txt # Python dependencies
├── setup.py # Package installation configuration
└── README.md # you are here
```

---

## Mathematical Foundations

### Backpropagation Example
For a linear layer $y = xW + b$:

| Gradient        | Formula                          |
|-----------------|----------------------------------|
| $\partial L/\partial W$ | $x^\top \cdot \partial L/\partial y$ |
| $\partial L/\partial b$ | $\sum(\partial L/\partial y)$     |
| $\partial L/\partial x$ | $\partial L/\partial y \cdot W^\top$ |

## Roadmap
- [x] Tensor operations
- [x] Linear layer
- [x] Backward pass
- [x] MSE loss
- [x] Broadcasting in backward pass
- [x] Working Linear Layer
- [ ] Activation functions (ReLU, Sigmoid)
- [ ] Optimizers (SGD, Adam)
- [ ] Convolutional layers
- [ ] More robust tests

---

> [!WARNING]
> **Disclaimer**: This is a toy project for learning purposes. For production use, consider established frameworks like PyTorch or TensorFlow.

## Key Components Breakdown

1. **Tensor Class (`tensor.py`)**
   - Core data structure tracking computational graph
   - Handles automatic differentiation via backward passes
   - Supports common operations (+, *, @, etc.) with operator overloading
   - Manages gradient computation and broadcasting

2. **Operations System**
   - `operation.py`: Base class for all operations
   - `arithmetic.py`: Elementary math operations with gradient rules
     - Add, Multiply, Power, Sum, Mean
   - `matmul.py`: Matrix multiplication with broadcasting support

3. **Neural Network Components**
   - `linear.py`: Fully-connected layer implementation
     - Xavier initialization for weights
     - Proper gradient tracking through matrix operations
   - `loss.py`: Mean Squared Error (MSE) implementation
     - Batch-aware gradient computation
     - Efficient computation graph construction

4. **Testing Infrastructure**
   - Gradient comparison tests against PyTorch
   - Detailed numerical validation
   - Rich terminal output for test results
   - Three test modes: arithmetic, mse, and all

---

## Performance Notes

- Optimized for learning and clarity over speed
- Memory-efficient tensor operations
- Automatic gradient cleanup
- Comparable performance to PyTorch for small to medium network