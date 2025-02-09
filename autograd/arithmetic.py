from autograd.operation import Operation
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
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return (
            _handle_broadcast(grad_output * b, a.shape),
            _handle_broadcast(grad_output * a, b.shape),
        )


class Divide(Operation):
    @staticmethod
    def forward(ctx, numerator, denominator):
        ctx.save_for_backward(numerator, denominator)
        return numerator / denominator

    @staticmethod
    def backward(ctx, grad_output):
        numerator, denominator = ctx.saved_tensors

        # numerator gradient with broadcasting
        num_grad = grad_output / denominator
        num_grad = _handle_broadcast(num_grad, numerator.shape)

        # denominator gradient with broadcasting
        den_grad = -grad_output * numerator / (denominator**2)
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
        ctx.input_shape = x.shape
        return np.mean(x)

    @staticmethod
    def backward(ctx, grad_output):
        # distributing gradient equally to all elements
        return np.full(ctx.input_shape, grad_output / np.prod(ctx.input_shape))


class Sum(Operation):
    @staticmethod
    def forward(ctx, x):
        ctx.input_shape = x.shape
        return np.sum(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient of sum is ones distributed over the input shape
        return np.full(ctx.input_shape, grad_output)


def _handle_broadcast(grad, target_shape):
    # adding missing dimensions
    padded_shape = (1,) * (grad.ndim - len(target_shape)) + target_shape
    # calculating reduction axes
    sum_axes = tuple(
        i for i in range(grad.ndim) if padded_shape[i] == 1 and grad.shape[i] > 1
    )
    return np.sum(grad, axis=sum_axes, keepdims=True).reshape(target_shape)
