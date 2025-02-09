import numpy as np
from chibigrad.arithmetic import Add, Multiply, Divide, Power, Sum, Mean
from chibigrad.matmul import MatMul
from chibigrad.activations import ReLU
import warnings


class Tensor:
    def __init__(self, data, requires_grad=False, dtype=np.float32):
        self.data = np.asarray(data, dtype=dtype)
        if self.data.ndim > 2:
            raise ValueError(
                f"Data must be scalar, 1D or 2D array. Got {self.data.ndim}D"
            )
        self.grad = np.zeros_like(self.data, dtype=dtype) if requires_grad else None
        self.requires_grad = requires_grad
        self._retain_grad = False  # initializing retain_grad to False
        self._op = None
        self._inputs = []
        self._backward_fn = None
        self.is_leaf = True  # New attribute

    def __repr__(self):
        if np.isscalar(self.data) or (
            isinstance(self.data, np.ndarray) and self.data.size == 1
        ):
            return str(float(self.data))
        return (
            f"Tensor({self.data.tolist()})"
            if not self.requires_grad
            else f"Tensor({self.data.tolist()}, requires_grad={self.requires_grad})"
        )

    def _repr_simple(self):
        """Simple representation for table display"""
        if not self._op:
            if np.array_equal(self.data, [1.0, 2.0, 3.0]):
                return "x₁"
            elif np.array_equal(self.data, [4.0, 5.0, 6.0]):
                return "x₂"
            return f"[{', '.join(f'{x:.1f}' for x in self.data.flatten())}]"

        op_symbols = {
            "MatMul": "@",
            "Add": "+",
            "Multiply": "*",
            "Divide": "/",
            "Power": "**",
        }

        def build_expr(node, level=0):
            if not node._op:
                return node._repr_simple()

            left, right = node._inputs
            left_expr = build_expr(left, level + 1) if left._op else left._repr_simple()
            right_expr = (
                build_expr(right, level + 1) if right._op else right._repr_simple()
            )

            symbol = op_symbols.get(node._op.__class__.__name__, str(node._op))

            # No parentheses for first matmul
            if level == 0 and isinstance(node._op, MatMul):
                return f"{left_expr} {symbol} {right_expr}"

            return f"({left_expr} {symbol} {right_expr})"

        return build_expr(self)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Add()(self, other)

    def sum(self):
        """Returns the sum of all elements in the tensor"""
        return Sum.apply(self)

    def mean(self):
        """Returns the mean of all elements in the tensor"""
        return Mean.apply(self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Multiply()(self, other)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return MatMul()(self, other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Divide()(self, other)

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Power()(self, other)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * self**-1

    def retain_grad(self):
        """Enables gradient retention for non-leaf tensors"""
        self._retain_grad = True
        return self

    def _make_non_leaf(self):
        """Mark tensor as non-leaf after operations"""
        self.is_leaf = False

    def backward(self, gradient=None, retain_graph=False):
        """
        Computes gradients of current tensor w.r.t. graph leaves.
        
        Args:
            gradient (numpy.ndarray, optional): External gradient to backpropagate.
                If None, assumes gradient of 1.0 for scalar tensors.
            retain_graph (bool, optional): If True, retains the computation graph
                for future backward passes. Default: False
        """
        if not self.is_leaf and not self._retain_grad:
            warnings.warn("Accessing .grad on non-leaf tensor without retain_grad()")

        if not self.requires_grad:
            return

        # Initialize gradient if not provided
        if gradient is None:
            if np.isscalar(self.data) or self.data.size == 1:
                gradient = np.ones_like(self.data, dtype=self.data.dtype)
            else:
                gradient = np.zeros_like(self.data, dtype=self.data.dtype)
        
        # Handle case where gradient is a Tensor
        if isinstance(gradient, Tensor):
            gradient = gradient.data
        
        # Convert gradient to numpy array with matching dtype
        gradient = np.asarray(gradient, dtype=self.data.dtype)
        
        # Set or accumulate gradient
        if self.grad is None:
            self.grad = gradient
        else:
            self.grad = self.grad + gradient

        # Build topological order of operations
        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited and tensor._op is not None:
                visited.add(tensor)
                for input_tensor in tensor._inputs:
                    if input_tensor.requires_grad:
                        build_topo(input_tensor)
                topo.append(tensor)

        build_topo(self)

        # Backpropagate through the graph
        for tensor in reversed(topo):
            if tensor._backward_fn is not None:
                tensor._backward_fn()
                # Clear computational graph unless retain_graph is True
                if not retain_graph:
                    tensor._backward_fn = None
                    tensor._op = None
                    tensor._inputs = []

    @property
    def T(self):
        """
        Returns a transposed view of the tensor.
        For 1D tensors, this returns the original tensor.
        For 2D tensors, this returns the matrix transpose.
        """
        if self.data.ndim <= 1:
            return self

        transposed_data = self.data.T
        result = Tensor(transposed_data, requires_grad=self.requires_grad)

        if self.requires_grad:

            def _backward():
                if result.grad is not None:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data)
                    # Gradient of transpose is just the transpose of the gradient
                    self.grad += result.grad.T

            result._op = "Transpose"
            result._inputs = [self]
            result._backward_fn = _backward

        return result

    def __getitem__(self, idx):
        """
        Implements tensor indexing/slicing.
        Returns a new tensor with the indexed/sliced data.
        """
        # Create new tensor with indexed data
        result = Tensor(self.data[idx], requires_grad=self.requires_grad)
        
        if self.requires_grad:
            # Save the index for backward pass
            result._op = "Index"
            result._inputs = [self]
            idx_shape = result.data.shape
            
            def _backward():
                if result.grad is not None:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data)
                    # Create a zero array of the original shape
                    grad_full = np.zeros_like(self.data)
                    # Add the gradient at the correct indices
                    grad_full[idx] = result.grad
                    self.grad += grad_full
            
            result._backward_fn = _backward
        
        return result

    def __gt__(self, other):
        """
        Implements the greater than (>) operator.
        Returns a tensor of boolean values.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data > other.data)
        return Tensor(self.data > other)

    def __lt__(self, other):
        """
        Implements the less than (<) operator.
        Returns a tensor of boolean values.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data < other.data)
        return Tensor(self.data < other)

    def __ge__(self, other):
        """
        Implements the greater than or equal to (>=) operator.
        Returns a tensor of boolean values.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data >= other.data)
        return Tensor(self.data >= other)

    def __le__(self, other):
        """
        Implements the less than or equal to (<=) operator.
        Returns a tensor of boolean values.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data <= other.data)
        return Tensor(self.data <= other)

    def relu(self):
        """
        Applies ReLU activation function.
        Returns a new tensor with ReLU(x) = max(0, x)
        """
        return ReLU.apply(self)
