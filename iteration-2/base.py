from context import Context


class Operation:
    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        from tensor import Tensor  # local import to break circular dependency

        # creating output tensor
        ctx = Context()
        output = Tensor(self.forward(
            ctx, *[arg.data for arg in args], **kwargs))

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
