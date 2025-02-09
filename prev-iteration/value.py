import numpy as np


class Value:
    """
    A Value object represents a single scalar value in the computation graph.
    It stores the value itself, its gradient, and the operation that produced
    it. Currently, this class supports arithmetic operations
    (addition, subtraction, multiplication, division) and provides methods
    for backpropagation.

    Further note to myself: Subtraction operation can be optimized to act like
    addition operations, so I can just get rid of the backward pass for it
    since the addition backward pass calculates the gradient correctly by
    virtue of the __neg__ method. However, for the sake of learning and
    creating a visual computation graph using the '-' symbol, I am not doing
    those at the moment. In the future, sure.
    """

    def __init__(self, data, _op="", label="", dtype=np.float32):
        self.data = np.asarray(data, dtype=dtype)
        if self.data.ndim > 2:
            raise ValueError(f"Data must be scalar, 1D or 2D array. Got {self.data.ndim}D")
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self._op = _op
        self._inputs = []
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        """Addition"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, "+", [self, other])
        out._inputs = [self, other]
        return out

    def __neg__(self):
        """Negation operator - returns new Value multiplied by -1"""
        return self * -1

    def __sub__(self, other):
        """Subtraction operator - implemented as a + (-b)"""
        return self + (-other)

    def __mul__(self, other):
        """scalar multiplication"""
        other = other if isinstance(other, Value) else Value(other)
        out_data = self.data * other.data
        result = Value(out_data, "*", [self, other])
        result._inputs = [self, other]
        return result

    def __rmul__(self, other):
        """
        This is a special method that allows us to multiply a `Value` object
        from the right side. For example, it lets us write `2 * x` where `x`
        is a `Value`, instead of just `x * 2`. This is part of Python's special
        method protocol - when Python sees `2 * x`, it first tries to call
        `2.__mul__(x)`, and if that fails (which it will since numbers don't
        know about our `Value` class), it falls back to calling
        `x.__rmul__(2)`. This is known as the "reflected" or "reversed"
        multiplication operation.
        """
        return self * other

    def __matmul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        result = Value(self.data @ other.data, _op="@")
        result._inputs = [self, other]
        return result

    def __pow__(self, exponent):
        """
        Implements power operation for Value objects.
        Handles both Value objects and scalar exponents.

        ```
        x ** y = x^y
        ```
        """
        if not isinstance(exponent, (int, float)):
            raise ValueError("Exponent must be a scalar")
        result = Value(self.data**exponent, _op="**")
        result._inputs = [self, Value(exponent)]
        return result

    def __radd__(self, other):
        """
        This is a special method that allows us to add a `Value` object from
        the right side. For example, it lets us write `2 + x` where `x` is a
        `Value`, instead of just `x + 2`. This is part of Python's special
        method protocol - when Python sees `2 + x`, it first tries to call
        `2.__add__(x)`, and if that fails (which it will since numbers don't
        know about our `Value` class), it falls back to calling
        `x.__radd__(2)`. This is known as the "reflected" or "reversed"
        addition operation.
        """
        return self + other

    def __truediv__(self, other):
        """
        This is a special method that implements division for `Value` objects.
        When Python sees an expression like `a / b` where `a` is a `Value`
        object, it calls this method. This method:
        1. Converts the divisor to a `Value` object if it isn't one already
        2. Creates a new `Value` object with the quotient of the data values
        3. Records the division operation & input values for use in backpropagation

        For example:

        ```python
        x = Value(8)
        y = Value(2)
        z = x / y  # z will be Value(4.0)
        ```

        The division operation is important for neural networks as it's used
        in operations like normalization and computing averages. During
        backpropagation, the gradient computation for division follows the
        quotient rule from calculus.
        """
        if not isinstance(other, Value):
            other = Value(other, label=str(other))
        result = Value(self.data / other.data, _op="/")
        result._inputs = [self, other]
        return result

    def __rtruediv__(self, other):
        """
        This is a special method that allows us to divide a `Value` object
        from the right side. For example, it lets us write `2 / x` where `x`
        is a `Value`, instead of just `x / 2`. This is part of Python's
        special method protocol - when Python sees `2 / x`, it first tries to
        call `2.__truediv__(x)`, and if that fails (which it will since
        numbers don't know about our `Value` class), it falls back to calling
        `x.__rtruediv__(2)`. This is known as the "reflected" or "reversed"
        division operation. Note that for division, the order matters - `a/b`
        is not the same as `b/a`, which is why we flip the operands here.
        """
        if not isinstance(other, Value):
            other = Value(other, label=str(other))
        result = Value(other.data / self.data, _op="/")
        result._inputs = [other, self]
        return result

    def mean(self):
        result = Value(np.mean(self.data), _op="mean")
        result._inputs = [self]
        return result

    def backward(self):
        # building topological order of the computation graph
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._inputs:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # zeroing gradients before backward pass
        for node in topo:
            node.grad = np.zeros_like(node.grad)
            
        self.grad = np.ones_like(self.data)

        # backpropagating gradients
        for v in reversed(topo):
            if v._op == "+":
                a, b = v._inputs
                grad = v.grad

                # Calculate broadcast axes for 'a'
                a_shape = a.data.shape
                padded_a_shape = (1,) * (grad.ndim - a.data.ndim) + a_shape
                sum_axes_a = tuple(
                    i
                    for i in range(grad.ndim)
                    if padded_a_shape[i] == 1 and grad.shape[i] > 1
                )
                a_grad = np.sum(grad, axis=sum_axes_a, keepdims=True).reshape(a_shape)

                # Calculate broadcast axes for 'b'
                padded_b_shape = (1,) * (grad.ndim - b.data.ndim) + b.data.shape
                sum_axes_b = tuple(
                    i
                    for i in range(grad.ndim)
                    if padded_b_shape[i] == 1 and grad.shape[i] > 1
                )
                b_grad = np.sum(grad, axis=sum_axes_b, keepdims=True).reshape(
                    b.data.shape
                )

                a.grad += a_grad
                b.grad += b_grad

            elif v._op == "@":
                a, b = v._inputs
                # improved batch matrix multiplication gradient
                a.grad += np.einsum('...ij,...jk->...ik', v.grad, b.data.swapaxes(-1, -2))
                b.grad += np.einsum('...ki,...kj->...ij', a.data, v.grad)

            elif v._op == "*":
                a, b = v._inputs
                grad = v.grad

                # Broadcast-aware gradient for 'a'
                a_grad = grad * b.data
                if a.data.ndim < grad.ndim:
                    sum_axes = tuple(range(grad.ndim - a.data.ndim))
                    a_grad = np.sum(a_grad, axis=sum_axes)
                a.grad += a_grad.reshape(a.data.shape)

                # Broadcast-aware gradient for 'b'
                b_grad = grad * a.data
                if b.data.ndim < grad.ndim:
                    sum_axes = tuple(range(grad.ndim - b.data.ndim))
                    b_grad = np.sum(b_grad, axis=sum_axes)
                b.grad += b_grad.reshape(b.data.shape)

            elif v._op == "**":
                base, exponent = v._inputs
                # Base gradient
                base_grad = exponent.data * (base.data ** (exponent.data - 1)) * v.grad
                if base.data.ndim < base_grad.ndim:
                    sum_axes = tuple(range(base_grad.ndim - base.data.ndim))
                    base_grad = np.sum(base_grad, axis=sum_axes)
                base.grad += base_grad.reshape(base.data.shape)

                # Exponent gradient
                log_term = np.log(np.maximum(base.data, 1e-8))
                exponent_grad = (base.data**exponent.data) * log_term * v.grad
                if exponent.data.ndim < exponent_grad.ndim:
                    sum_axes = tuple(range(exponent_grad.ndim - exponent.data.ndim))
                    exponent_grad = np.sum(exponent_grad, axis=sum_axes)
                exponent.grad += exponent_grad.reshape(exponent.data.shape)

            elif v._op == "/":
                numerator, denominator = v._inputs
                # Numerator gradient
                num_grad = v.grad / denominator.data
                if numerator.data.ndim < num_grad.ndim:
                    sum_axes = tuple(range(num_grad.ndim - numerator.data.ndim))
                    num_grad = np.sum(num_grad, axis=sum_axes)
                numerator.grad += num_grad.reshape(numerator.data.shape)

                # Denominator gradient
                den_grad = -v.grad * numerator.data / (denominator.data**2)
                if denominator.data.ndim < den_grad.ndim:
                    sum_axes = tuple(range(den_grad.ndim - denominator.data.ndim))
                    den_grad = np.sum(den_grad, axis=sum_axes)
                denominator.grad += den_grad.reshape(denominator.data.shape)

            elif v._op == "mean":
                (a,) = v._inputs
                # Distribute gradient according to original input shape
                grad = v.grad / np.prod(a.data.shape)
                a.grad += np.full(a.data.shape, grad)
