from chibigrad.operation import Operation
import numpy as np


class Add(Operation):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _handle_broadcast(grad_output, ctx.a_shape),
            _handle_broadcast(grad_output, ctx.b_shape),
        )


class Multiply(Operation):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        ctx.same_tensor = a is b  # Check if same tensor is used twice
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        
        if ctx.same_tensor:
            # If x * x, the gradient is 2x * grad_output
            return _handle_broadcast(2 * a * grad_output, a.shape)
        
        # Normal case for different tensors
        return (
            _handle_broadcast(grad_output * b, a.shape),
            _handle_broadcast(grad_output * a, b.shape),
        )


class Divide(Operation):
    @staticmethod
    def forward(ctx, numerator, denominator):
        ctx.save_for_backward(numerator, denominator)
        # Handle division by zero gracefully
        result = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        numerator, denominator = ctx.saved_tensors
        
        # Handle division by zero in gradients
        safe_denominator = np.where(denominator == 0, np.ones_like(denominator), denominator)
        
        # numerator gradient
        num_grad = grad_output / safe_denominator
        num_grad = np.where(denominator == 0, 0, num_grad)  # Zero gradient where undefined
        num_grad = _handle_broadcast(num_grad, numerator.shape)
        
        # denominator gradient
        den_grad = -grad_output * numerator / (safe_denominator**2)
        den_grad = np.where(denominator == 0, 0, den_grad)  # Zero gradient where undefined
        den_grad = _handle_broadcast(den_grad, denominator.shape)
        
        return num_grad, den_grad


class Power(Operation):
    @staticmethod
    def forward(ctx, base, exponent):
        ctx.save_for_backward(base, exponent)
        return np.power(base, exponent)

    @staticmethod
    def backward(ctx, grad_output):
        base, exponent = ctx.saved_tensors

        # base gradient with broadcasting handling
        base_grad = exponent * np.power(base, exponent - 1) * grad_output
        base_grad = _handle_broadcast(base_grad, base.shape)

        # exponent gradient with numerical stability
        log_term = np.log(np.maximum(base, 1e-8))
        exponent_grad = np.power(base, exponent) * log_term * grad_output
        exponent_grad = _handle_broadcast(exponent_grad, exponent.shape)

        return base_grad, exponent_grad


class Mean(Operation):
    @staticmethod
    def forward(ctx, x):
        """Forward pass of mean operation"""
        ctx.save_for_backward(x)
        ctx.input_shape = x.shape
        ctx.n_elements = np.prod(x.shape)
        
        # Handle NaN values
        if np.any(np.isnan(x)):
            return np.nan
        return np.mean(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of mean operation"""
        x, = ctx.saved_tensors
        
        # If input had NaN, propagate NaN in gradient
        if np.any(np.isnan(x)):
            return np.full(ctx.input_shape, np.nan, dtype=x.dtype)
        
        # Normal case
        grad = np.full(ctx.input_shape, grad_output) / ctx.n_elements
        return grad.astype(x.dtype)


class Sum(Operation):
    @staticmethod
    def forward(ctx, x):
        ctx.input_shape = x.shape
        result = np.sum(x)
        print(f"Sum forward - Input shape: {x.shape}, Output: {result}")
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # For sum operation, the gradient is distributed equally to all input elements
        # grad_output is a scalar, we need to broadcast it to all elements
        grad = np.ones(ctx.input_shape) * grad_output
        print(f"Sum backward - Input grad: {grad_output}, Output grad shape: {grad.shape}")
        return grad


def _handle_broadcast(grad, target_shape):
    # adding missing dimensions
    padded_shape = (1,) * (grad.ndim - len(target_shape)) + target_shape
    # calculating reduction axes
    sum_axes = tuple(
        i for i in range(grad.ndim) if padded_shape[i] == 1 and grad.shape[i] > 1
    )
    return np.sum(grad, axis=sum_axes, keepdims=True).reshape(target_shape)
