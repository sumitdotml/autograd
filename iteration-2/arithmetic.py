from base import Operation
import numpy as np

class Add(Operation):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
        return a + b
    
    @staticmethod
    def backward(ctx, grad_output):
        # Replicate the broadcasting logic from Value class
        def handle_broadcast(grad, shape):
            # Add missing dimensions
            padded_shape = (1,) * (grad.ndim - len(shape)) + shape
            # Calculate reduction axes
            sum_axes = tuple(
                i for i in range(grad.ndim)
                if padded_shape[i] == 1 and grad.shape[i] > 1
            )
            # Reduce and reshape
            return np.sum(grad, axis=sum_axes, keepdims=True).reshape(shape)
        
        return (handle_broadcast(grad_output, ctx.a_shape),
                handle_broadcast(grad_output, ctx.b_shape))

class Multiply(Operation):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        
        # Handle broadcasting for a's gradient
        a_grad = grad_output * b
        if len(a.shape) < len(grad_output.shape):
            sum_axes = tuple(range(len(grad_output.shape) - len(a.shape)))
            a_grad = np.sum(a_grad, axis=sum_axes)
            
        # Handle broadcasting for b's gradient
        b_grad = grad_output * a
        if len(b.shape) < len(grad_output.shape):
            sum_axes = tuple(range(len(grad_output.shape) - len(b.shape)))
            b_grad = np.sum(b_grad, axis=sum_axes)
            
        return a_grad.reshape(a.shape), b_grad.reshape(b.shape)

class Divide(Operation):
    @staticmethod
    def forward(ctx, numerator, denominator):
        ctx.save_for_backward(numerator, denominator)
        return numerator / denominator
    
    @staticmethod
    def backward(ctx, grad_output):
        numerator, denominator = ctx.saved_tensors
        
        # Numerator gradient with broadcasting
        num_grad = grad_output / denominator
        num_grad = _handle_broadcast(num_grad, numerator.shape)
        
        # Denominator gradient with broadcasting
        den_grad = -grad_output * numerator / (denominator ** 2)
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
        
        # Base gradient with broadcasting handling
        base_grad = exponent * np.power(base, exponent - 1) * grad_output
        base_grad = _handle_broadcast(base_grad, base.shape)
        
        # Exponent gradient with numerical stability
        log_term = np.log(np.maximum(base, 1e-8))
        exponent_grad = np.power(base, exponent) * log_term * grad_output
        exponent_grad = _handle_broadcast(exponent_grad, exponent.shape)
        
        return base_grad, exponent_grad
    

def _handle_broadcast(grad, target_shape):
    # Add missing dimensions
    padded_shape = (1,) * (grad.ndim - len(target_shape)) + target_shape
    # Calculate reduction axes
    sum_axes = tuple(i for i in range(grad.ndim) if padded_shape[i] == 1 and grad.shape[i] > 1)
    return np.sum(grad, axis=sum_axes, keepdims=True).reshape(target_shape)