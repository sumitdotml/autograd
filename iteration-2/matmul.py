from base import Operation
import numpy as np

class MatMul(Operation):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return np.matmul(a, b)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        
        # Handle vector dot product case
        if a.ndim == 1 and b.ndim == 1:
            a_grad = grad_output * b
            b_grad = grad_output * a
            return a_grad, b_grad
            
        # Handle matrix/vector cases
        a_2d = a if a.ndim >= 2 else a.reshape(-1, 1)
        b_2d = b if b.ndim >= 2 else b.reshape(1, -1)
        grad_2d = grad_output.reshape(-1, 1) if grad_output.ndim < 2 else grad_output

        try:
            a_grad = np.matmul(grad_2d, b_2d.T)
            b_grad = np.matmul(a_2d.T, grad_2d)
        except ValueError as e:
            raise ValueError(f"MatMul backward error: {str(e)}\n"
                             f"Shapes: grad_output={grad_output.shape}, "
                             f"b_2d.T={b_2d.T.shape}, a_2d.T={a_2d.T.shape}") from e

        # Restore original dimensions
        if a.ndim == 1:
            a_grad = a_grad.squeeze(0)
        if b.ndim == 1:
            b_grad = b_grad.squeeze(-1)
            
        return a_grad, b_grad