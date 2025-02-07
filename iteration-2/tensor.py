import numpy as np
from arithmetic import Add, Multiply, Divide, Power
from matmul import MatMul


class Tensor:
    def __init__(self, data, requires_grad=False, dtype=np.float32):
        self.data = np.asarray(data, dtype=dtype)
        if self.data.ndim > 2:
            raise ValueError(
                f"Data must be scalar, 1D or 2D array. Got {self.data.ndim}D"
            )
        self.grad = np.zeros_like(
            self.data, dtype=dtype) if requires_grad else None
        self.requires_grad = requires_grad
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

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Add()(self, other)

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

    def backward(self):
        if not self.requires_grad:
            return

        # zeroing gradients before backward pass
        for node in self._get_all_nodes():
            if node.requires_grad and node.grad is not None:
                node.grad.fill(0)

        # initializing self gradient to 1
        self.grad = np.ones_like(self.data)

        # building topological order
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._inputs:
                    if child.requires_grad:
                        build_topo(child)
                topo.append(node)

        build_topo(self)

        # backpropagating through the graph
        for node in reversed(topo):
            if node._backward_fn is not None:
                node._backward_fn()

    def _get_all_nodes(self):
        visited = set()
        nodes = []

        def _traverse(node):
            if node not in visited:
                visited.add(node)
                for child in node._inputs:
                    _traverse(child)
                nodes.append(node)

        _traverse(self)
        return nodes
