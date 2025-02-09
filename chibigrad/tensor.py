import numpy as np
from chibigrad.arithmetic import Add, Multiply, Divide, Power, Sum, Mean
from chibigrad.matmul import MatMul


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

    def backward(self):
        """
        Computes gradients of current tensor w.r.t. graph leaves.
        """
        # If no grad_fn, this is a leaf
        if not self._backward_fn:
            return

        # initializing gradient if not already set
        if self.grad is None:
            # for scalar outputs, initializing with 1.0
            # for non-scalar outputs, we need all ones
            self.grad = np.ones_like(self.data)

        # getting nodes in topological order
        topo = self._build_topo()

        # Go one variable at a time and apply the chain rule
        for node in reversed(topo):
            if node._backward_fn:
                node._backward_fn()

    def _build_topo(self):
        visited = set()
        topo = []

        def _traverse(node):
            if node not in visited:
                visited.add(node)
                for child in node._inputs:
                    _traverse(child)
                topo.append(node)

        _traverse(self)
        return topo

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
