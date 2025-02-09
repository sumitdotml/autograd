# Autograd Engine from Scratch

A project where I focus on understanding how tensors fit into the larger autograd system and how they enable automatic differentiation.

> [!NOTE]
> __Status:__ In Progress
---

I have currently managed to implement the basic tensor operations, the linear layer and the backward pass for basic operations like addition, subtraction, multiplication and division. Also improved it to accept both scalars and numpy arrays as input. Also added a simple loss function (MSE).

You can go inside the `autograd` directory and play around with the [check.py](./autograd/check.py) file where I compare and contrast some simple arithmetic with both my implementation and that of PyTorch.

---

## Update Logs

#### Iteration 2: autograd
- 2025/02/07: Modularized the code by breaking down the Value class into smaller classes for `Tensor`, `Operation` and `Module`. Created the `Linear` class that supports `forward` and `backward` pass. Created `arithmetic` and `matmul` classes that support the operations between two tensors. Created a directory [autograd](./autograd/) for the code.

--- Iteration 2 End ---

#### Iteration 1 Beginning

- 2025/02/06: Managed to fix the broadcasting issue. Tried tallying the forward pass as well as the gradients of `W` and `b` during backprop with PyTorch: forward pass seems to be correct. There is a significant difference in the `W` and `b` gradients (check [notebook](./iteration-1/notebook.ipynb) for details) compared to that of PyTorch, so going to see what I've done wrong there.</br> <u>**Update**: managed to fix it</u>.

- 2025/02/01: Currently facing an issue with broadcasting in the backward pass for the linear layer  ever since I changed the Value class to support not only scalars but also numpy arrays. Will try to fix this tomorrow.

---

## Progress Tracker

#### --- Iteration 1 ---

- [x] Tensor operations
- [x] Linear layer
- [x] Backward pass
- [x] MSE loss
- [x] Broadcasting in backward pass
- [x] Done: Found significant gradient difference issue while testing the same data with PyTorch, currently trying to fix it
- [x] Numerical stability improvements (like `log_term = np.log(np.maximum(base_term, 1e-10))` instead of `log_term = np.log(base_data + 1e-10)`)

--- Iteration 1 End ---

#### --- Iteration 2 ---
- [x] Modularize the code by breaking down the Value class into smaller classes for `Tensor`, `Operation` and `Module`.
- [ ] Add activation functions (ReLU and Sigmoid for now)
- [ ] Add optimizer implementations (SGD, Adam, etc.)
- [ ] Add more tensor operations (like `matmul`, `cross_entropy`, etc.)
- [ ] Add sequential container (like `nn.Sequential`) maybe
- [ ] Do lot of testing such as:
   - [ ] Gradient checking
   - [ ] Compare the outputs of this implementation with PyTorch for the same data
   - [ ] Unit tests for all operations
   - [ ] More unit tests for edge cases
   - [ ] Test the implementation with a simple neural network
   and so on.
- [ ] Make the `iteration-2` directory the default one

(will add more as I go along)

Trying my best to not touch PyTorch and implement everything using pure numpy and basic python.

---

## Project Overview
This project implements a minimal autograd engine capable of automatic differentiation. The engine will have to:
- Support basic tensor operations
- Build and traverse computational graphs
- Compute gradients via backpropagation
- Implement a linear layer and mean squared error loss

## Implementation Details

### Core Components
1. **Tensor Class**
   - Stores data and gradients
   - Tracks operations that created it
   - Implements basic arithmetic operations

2. **Operation Base Class**
   - Defines forward and backward methods
   - Tracks input tensors
   - Manages computational graph

3. **Linear Layer**
   - Implements y = xW + b
   - Stores weight and bias parameters
   - Computes gradients for W, b, and x

4. **Mean Squared Error**
   - Implements L = 1/n * Σ(y_pred - y_true)²
   - Computes gradients w.r.t. predictions

### Computational Graph
- Built during forward pass
- Nodes represent operations
- Edges represent tensors
- Traversed in reverse during backpropagation

## Mathematical Foundations

### Linear Layer
1. Forward Pass:
   $$
   y = xW + b
   $$

2. Backward Pass:

$$
\frac{\partial L}{\partial W} = x^\top \frac{\partial L}{\partial y}   
$$
$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial y}
$$
$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} W^\top
$$

In the above equations, $x$ is the input, $W$ is the weight, $b$ is the bias, $y$ is the output, and $L$ is the loss.

Explanation:
- $\frac{\partial L}{\partial W}$ is the gradient of the loss with respect to the weight.
- $\frac{\partial L}{\partial b}$ is the gradient of the loss with respect to the bias.
- $\frac{\partial L}{\partial x}$ is the gradient of the loss with respect to the input.
- $W^\top$ is the transpose of the weight matrix.
- $\frac{\partial L}{\partial y}$ is the gradient of the loss with respect to the output.
- This is a chain rule, meaning: $dL/dx = dL/dy * dy/dx$. In our case, $dL/dx = dL/dy * W$ because $y = xW + b$.

### Mean Squared Error
1. Forward Pass:
   $$
   L = 1/n * Σ(y_pred - y_true)²
   $$
2. Backward Pass:
   $$
   ∂L/∂y_pred = 2/n * (y_pred - y_true)
   $$

## Usage Example

```python
# Create tensors
x = Tensor([[1.0, 2.0], [3.0, 4.0]])
W = Tensor([[0.1, 0.2], [0.3, 0.4]])
b = Tensor([0.5, 0.6])

# Create linear layer
linear = Linear(2, 2)
linear.W = W
linear.b = b

# Forward pass
y = linear(x)

# Compute loss
y_true = Tensor([[1.0, 1.0], [1.0, 1.0]])
loss = MSELoss()(y, y_true)

# Backward pass
loss.backward()

# Print gradients
print(linear.W.grad)
print(linear.b.grad)
```

## Testing and Verification
1. **Numerical Gradient Checking**
   - Compare computed gradients with finite differences
   - Verify correctness of backpropagation

2. **Comparison with PyTorch**
   - Implement same model in PyTorch
   - Compare outputs and gradients

3. **Unit Tests**
   - Test individual operations
   - Test gradient computations
   - Test edge cases

## Future Extensions
1. Add more operations (e.g., ReLU, Softmax)
2. Implement cross-entropy loss
3. Add optimization algorithms (SGD, Adam)
4. Support GPU acceleration
5. Add visualization of computational graph
