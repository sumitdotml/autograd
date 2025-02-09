from chibigrad.context import Context
import numpy as np


class Operation:
    @classmethod
    def apply(cls, *args):
        """
        Creates a new tensor and sets up the backward pass
        """
        # creating operation instance of the specific class (Sum, Add, etc.)
        op = cls()

        # Lazy import to avoid circular dependency
        from chibigrad.tensor import Tensor

        # converting args to Tensors if they aren't already
        tensor_args = [arg if isinstance(arg, Tensor) else Tensor(arg) for arg in args]

        # getting the requires_grad flag - True if any input requires grad
        requires_grad = any(t.requires_grad for t in tensor_args)

        # Forward pass
        ctx = Context()
        result_data = op.forward(ctx, *[t.data for t in tensor_args])
        result = Tensor(result_data, requires_grad=requires_grad)

        # setting up backward pass if needed
        if requires_grad:
            result._op = op
            result._inputs = tensor_args

            def _backward():
                if result.grad is None:
                    result.grad = np.ones_like(result.data)
                grads = op.backward(ctx, result.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                for t, g in zip(tensor_args, grads):
                    if t.requires_grad:
                        if t.grad is None:
                            t.grad = np.zeros_like(t.data)
                        t.grad += g

            result._backward_fn = _backward

        return result

    @staticmethod
    def forward(ctx, *args):
        """Forward pass computation"""
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass computation"""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        from chibigrad.tensor import Tensor  # local import to break circular dependency

        # creating output tensor
        ctx = Context()
        output = Tensor(self.forward(ctx, *[arg.data for arg in args], **kwargs))

        if any(arg.requires_grad for arg in args):
            output.requires_grad = True
            output._op = self
            output._inputs = args

            # saving for backward
            def _backward_fn():
                grads = self.backward(ctx, output.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                for arg, grad in zip(args, grads):
                    if arg.requires_grad:
                        if arg.grad is None:
                            arg.grad = grad
                        else:
                            arg.grad += grad

            output._backward_fn = _backward_fn

        return output
